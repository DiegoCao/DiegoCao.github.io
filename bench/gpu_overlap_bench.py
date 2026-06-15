"""
GPU overlap benchmark — quantifies the Ampere `cp.async` win discussed in
the blog post "How GPU Matmul Got Asynchronous" by sweeping Triton's
`num_stages` parameter on a single GEMM.

What it actually measures
-------------------------
The exact same Triton GEMM kernel is launched with progressively deeper
software pipelines. `num_stages` in Triton controls how many iterations'
worth of `cp.async` loads stay in flight simultaneously:

    num_stages = 1   →  no pipelining (Pre-Ampere style: load → sync → compute)
    num_stages = 2   →  double-buffered (basic Ampere pipeline)
    num_stages = 3+  →  deeper pipeline (production cp.async overlap)

Because everything else is identical (kernel, dtype, tile shape, hardware),
the delta between configurations isolates the impact of in-CTA async-load
overlap. On Ampere+ GPUs the gap should be 2–3× at typical LLM shapes.

On Hopper (compute capability ≥ 9.0) Triton additionally selects WGMMA
(`wgmma.mma_async`) on its own when tile sizes are large enough, so the
same script picks up the Hopper story for free without extra knobs.

Requirements
------------
    pip install torch triton
    NVIDIA GPU with compute capability ≥ 7.5 (Turing+); Ampere or newer
    recommended to see the pipelining effect.

Run
---
    python bench/gpu_overlap_bench.py
    python bench/gpu_overlap_bench.py --m 8192 --n 8192 --k 8192
    python bench/gpu_overlap_bench.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass

try:
    import torch
except ImportError:
    print("error: torch is not installed. `pip install torch` first.", file=sys.stderr)
    sys.exit(1)

try:
    import triton
    import triton.language as tl
except ImportError:
    print("error: triton is not installed. `pip install triton` first.", file=sys.stderr)
    sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
# Triton GEMM kernel — single source of truth; `num_stages` is the only knob.
# ────────────────────────────────────────────────────────────────────────────
@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Threadblock-swizzled program ID for better L2 locality.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def _run(A, B, num_stages, num_warps, block_m, block_n, block_k, group_m):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_stages=num_stages, num_warps=num_warps,
    )
    return C


# ────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    num_stages: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    tflops: float
    speedup_vs_baseline: float
    correct: bool


def benchmark(M, N, K, num_stages_list, block_m=128, block_n=128, block_k=32,
              num_warps=4, group_m=8, warmup=20, repeat=100):
    if not torch.cuda.is_available():
        sys.exit("error: CUDA not available. This benchmark needs an NVIDIA GPU.")

    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    C_ref = (A.float() @ B.float()).half()  # higher-precision reference

    results: list[Result] = []
    baseline_ms = None

    for ns in num_stages_list:
        # Warmup.
        for _ in range(warmup):
            C = _run(A, B, ns, num_warps, block_m, block_n, block_k, group_m)
        torch.cuda.synchronize()

        # Correctness — within fp16 tolerance.
        correct = torch.allclose(C.float(), C_ref.float(), atol=1.0, rtol=5e-2)

        # Timing with CUDA events; per-iteration record/elapsed.
        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = _run(A, B, ns, num_warps, block_m, block_n, block_k, group_m)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = statistics.median(times)
        p10_ms = times[max(0, int(0.10 * len(times)) - 1)]
        p90_ms = times[min(len(times) - 1, int(0.90 * len(times)))]
        tflops = (2.0 * M * N * K) / (median_ms / 1000.0) / 1e12
        if baseline_ms is None:
            baseline_ms = median_ms
        speedup = baseline_ms / median_ms
        results.append(Result(ns, median_ms, p10_ms, p90_ms, tflops, speedup, correct))

    return results


# ────────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────────
def device_info():
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    return {
        "name": props.name,
        "compute_capability": cc,
        "sms": props.multi_processor_count,
        "memory_gb": round(props.total_memory / 1024 ** 3, 1),
        "torch": torch.__version__,
        "triton": triton.__version__,
    }


def arch_label(cc: str) -> str:
    major = int(cc.split(".")[0])
    return {
        7: "Volta/Turing (pre-cp.async)",
        8: "Ampere (cp.async)",
        9: "Hopper (TMA + WGMMA)",
        10: "Blackwell (tcgen05 + tensor memory)",
    }.get(major, f"compute capability {cc}")


def print_table(results, info, shape):
    M, N, K = shape
    print()
    print(f"GPU:         {info['name']}")
    print(f"Arch:        {arch_label(info['compute_capability'])}  "
          f"(cc {info['compute_capability']}, {info['sms']} SMs, {info['memory_gb']} GB)")
    print(f"Software:    torch {info['torch']} · triton {info['triton']}")
    print(f"GEMM:        M={M} N={N} K={K}  fp16  (FLOPs = {2*M*N*K/1e9:.1f}G)")
    print()
    print(f"  {'num_stages':>11}   {'median ms':>10}   {'p10 / p90 ms':>15}"
          f"   {'TFLOPS':>8}   {'speedup':>8}   ok")
    print(f"  {'-'*11}   {'-'*10}   {'-'*15}   {'-'*8}   {'-'*8}   --")
    for r in results:
        print(f"  {r.num_stages:>11}   {r.median_ms:>10.3f}   "
              f"{r.p10_ms:>5.3f} / {r.p90_ms:>6.3f}   "
              f"{r.tflops:>8.1f}   {r.speedup_vs_baseline:>7.2f}x   "
              f"{'✓' if r.correct else '✗'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--m", type=int, default=4096, help="M dimension (default 4096)")
    ap.add_argument("--n", type=int, default=4096, help="N dimension (default 4096)")
    ap.add_argument("--k", type=int, default=4096, help="K dimension (default 4096)")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="num_stages values to sweep (default: 1 2 3 4 5)")
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=32)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--json", type=str, default=None,
                    help="write structured results to this path")
    args = ap.parse_args()

    results = benchmark(
        args.m, args.n, args.k, args.stages,
        block_m=args.block_m, block_n=args.block_n, block_k=args.block_k,
        num_warps=args.num_warps,
        warmup=args.warmup, repeat=args.repeat,
    )
    info = device_info()
    print_table(results, info, (args.m, args.n, args.k))

    if args.json:
        payload = {
            "device": info,
            "shape": {"M": args.m, "N": args.n, "K": args.k, "dtype": "fp16"},
            "config": {
                "BLOCK_M": args.block_m, "BLOCK_N": args.block_n,
                "BLOCK_K": args.block_k, "num_warps": args.num_warps,
                "warmup": args.warmup, "repeat": args.repeat,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  → wrote {args.json}\n")


if __name__ == "__main__":
    main()
