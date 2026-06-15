# GPU overlap benchmark

Companion benchmark for the blog post **[How GPU Matmul Got Asynchronous](../blog/gpu-matmul-asynchronous.html)**. Quantifies the Ampere `cp.async` software-pipelining win by sweeping Triton's `num_stages` parameter on a single GEMM kernel.

## What it measures

The exact same Triton GEMM kernel is launched with progressively deeper software pipelines. Triton's `num_stages` controls how many iterations' worth of `cp.async` loads stay in flight simultaneously:

| `num_stages` | Behaviour | Generation analogue |
|---|---|---|
| 1 | No pipelining (load → sync → compute) | Pre-Ampere |
| 2 | Double-buffered cp.async pipeline | Basic Ampere |
| 3+ | Deeper pipeline (production setting) | Tuned Ampere / Hopper |

Because everything else — kernel, dtype, tile shape, hardware — is identical, the delta between configurations isolates the impact of in-CTA async-load overlap.

On Hopper (compute capability ≥ 9.0) Triton additionally selects `wgmma.mma_async` automatically when the tile shape supports it, so the same script picks up the Hopper compute side for free.

## Run

```bash
pip install -r bench/requirements.txt
python bench/gpu_overlap_bench.py
```

Defaults: 4096×4096×4096 fp16 GEMM, sweeps `num_stages ∈ {1, 2, 3, 4, 5}`, 20 warmup + 100 timed iterations, CUDA-event timing.

### Useful invocations

```bash
# Larger shape — more bandwidth-bound, bigger pipelining win
python bench/gpu_overlap_bench.py --m 8192 --n 8192 --k 8192

# Small-batch decode-like shape
python bench/gpu_overlap_bench.py --m 16 --n 14336 --k 4096

# Custom tile (the Hopper TMA path likes 128x256x64+)
python bench/gpu_overlap_bench.py --block-m 128 --block-n 256 --block-k 64 --num-warps 8

# Persist results for cross-machine comparison
python bench/gpu_overlap_bench.py --json h100.json
```

## Sample output

```
GPU:         NVIDIA H100 80GB HBM3
Arch:        Hopper (TMA + WGMMA)  (cc 9.0, 132 SMs, 79.6 GB)
Software:    torch 2.3.0 · triton 2.3.0
GEMM:        M=4096 N=4096 K=4096  fp16  (FLOPs = 137.4G)

   num_stages    median ms      p10 / p90 ms     TFLOPS    speedup   ok
  -----------   ----------   ---------------   --------   --------   --
            1        4.321     4.301 /  4.402       31.8      1.00x   ✓
            2        2.187     2.171 /  2.221       62.8      1.98x   ✓
            3        1.553     1.541 /  1.587       88.5      2.78x   ✓
            4        1.512     1.499 /  1.541       90.9      2.86x   ✓
            5        1.498     1.486 /  1.519       91.7      2.88x   ✓
```

(Numbers above are illustrative — fill in your own from a real run.)

## Hardware notes

| GPU | What you should see |
|---|---|
| **V100 / T4** (cc 7.x) | Little to no speedup — `cp.async` doesn't exist; Triton falls back to synchronous loads and `num_stages` is a no-op. This is itself the data point: pre-Ampere couldn't pipeline within a CTA. |
| **A100 / A6000** (cc 8.x) | Expect ~2–3× from `num_stages=1` → `num_stages=3+`. This is the Ampere win the blog post argues about, measured directly. |
| **H100 / H200** (cc 9.x) | Similar relative speedup on this script; Triton also auto-selects WGMMA for large tiles, so absolute TFLOPS will be much higher than A100. |
| **B100 / B200** (cc 10.x) | Same script runs but doesn't exercise `tcgen05` / tensor memory — that requires a different kernel path. PRs welcome. |

## Caveats

- This is a microbenchmark of a single isolated GEMM. Real kernel performance in CUTLASS / cuBLAS / vendor libraries is higher because they tile, schedule, and pipeline more aggressively than the simple Triton template above. The *delta* between `num_stages` configurations is the interesting signal, not the absolute TFLOPS number.
- Triton handles the `cp.async` ⇄ synchronous fallback transparently based on compute capability, which is convenient but means you can't actually run a "pre-Ampere style" kernel on an A100 by setting `num_stages=1` — what you get is a "no software pipelining" kernel, which is the moral equivalent. If you want true synchronous pre-Ampere behaviour for comparison, run this on a Turing GPU.
- The reference uses an fp32 accumulator (`A.float() @ B.float()`) compared against the fp16 Triton output with `atol=1.0, rtol=5e-2` — tight enough to catch real bugs, loose enough not to false-positive on rounding.

## Extending

Reasonable additions (PRs welcome):

- **Hopper-specific A/B**: same kernel forced through `wgmma.mma_async` vs `mma.sync` paths.
- **Persistent-kernel variant**: a long-running kernel that processes many tiles, demonstrating the launch-overhead win the blog post attributes to Hopper.
- **Memory-bound shape**: small-batch decode (e.g. 1×4096×14336) to highlight where Tensor Cores stop being the bottleneck.
- **CSV / Markdown emitter** for easy paste into the blog post.
