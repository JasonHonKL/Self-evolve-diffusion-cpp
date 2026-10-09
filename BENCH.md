# BENCH — measured on this box (8 cores, 23 GB RAM, no GPU)

Date: 2026-10-09 (issue #17). Machine context: measured fp32 ceiling **0.16 TFLOPS**
(README), SOTA reference **Ovi 5s/720p/50 steps = 83 s on one H100** (~990 TFLOPS).
All numbers below are from `make`-built binaries run from the repo root.

## Kernel benches (`./build/tests/test_core --bench`, threads=8)

| kernel | shape | time | throughput |
|---|---|---|---|
| matmul fp32 [openblas] | 1024x1024 x 1024x1024 | 19.40 ms | 110.7 GFLOPS |
| matmul fp32 [openblas] | 3072x5120 x 5120x5120 | 606.85 ms | 265.4 GFLOPS |
| matmul fp32 [custom] | 192x512 x 512x512 | 3.80 ms | 26.5 GFLOPS |
| matmul fp32 [custom] | 240x1024 x 1024x256 | 2.40 ms | 52.4 GFLOPS |
| **qgemm int8** | 3072x5120 @ 5120x5120 | 731.32 ms | **220.2 TOPS** |

fp32 peak measured 265 GFLOPS (0.27 TFLOPS, large OpenBLAS shapes); int8 qgemm
220 TOPS — ~830x under an H100's int8 tensor rate, but ~6.6x the fp32 path.

## T5 text encoder (`tests/test_ovit8.cpp` part 4 + `test_t5enc`)

Weights: 10.6 GB bf16 mmap (`ckpts/t5_enc.safetensors`).

| path | s/prompt |
|---|---|
| bf16 views, per-call bf16->f32 convert (test_t5enc, L=16) | 26–84 s |
| fp32 weight cache (issue #17), L=16 | 3.3–4.2 s |
| fp32 weight cache, L=512 (max text_len) | **42.8 s** |

fp32 cache: 241 tensors / 18.52 GB built once in **54.6 s** (parallel convert +
`madvise(MADV_DONTNEED)` on converted bf16 pages so peak RSS ≈ cache size).
Golden parity identical with and without cache (cosine 0.999176/0.999546/0.999695,
same digits). Pipeline MUST sequence T5 before the DiT: `encode -> free_weights()
-> load DiT` (18.5 GB cache + 12 GB pack > RAM).

## Ovi DiT int8 forward — first real-shape bench (`tests/test_ovit8.cpp` part 2)

Real config (30+30 layers, dim 3072, ffn 14336), int8 pack `ckpts/Ovi/model.sdcpp`
(12 GB, 2073 tensors). Inputs: video latent [48,4,8,8] -> **64 video tokens**,
audio [100,20], text ctx 64 -> padded 512. fp32 real bench skipped: the bf16
safetensors fp32 path would materialize ~46 GB of weights.

**Before the #17 perf work** (baseline for everything below):

| phase | time |
|---|---|
| pack load (mmap + dequant of non-gemm weights) | 6.4 s |
| forward iter 0 (cold: pack page-in) | 287.8 s |
| forward warm (best of 3) | **205.2 s** |

Warm breakdown: linear/gemm (qgemm) 25.7 s | sdpa 3.1 s | other 176.4 s.
"Other" is dominated by the audio ConvMLP patch embed — scalar cache-hostile
`conv1d` over w1/w2/w3 [8192,3072,7] (~70 GFLOP + ~200 GB strided weight traffic
per forward); runs once per forward. int8 parity at golden cfg: cosine
**0.9999 (out_video) / 0.9998 (out_audio)** vs the fp32 forward.

## #17 perf pass — three optimizations (measured before/after)

### 1. qgemm B-widening fix (`src/core/qgemm.cpp`)

Old policy widened the ENTIRE B panel [N,K] to int16 whenever M>=32 — at model
scale that streamed ~3x weight bytes per call (read i8 + write s16 + read s16
per forward, ~22-24 GB effective for 11 GB of weights). New policy: **never
widen below M=2048; above it, widen per column-chunk in task-local scratch**
(chunk=64 B rows = 640 KB, L2-resident; measured best of 64/128/256/512/1024/2048 —
128+ falls out of L2 and loses up to 4x at M=16384). Peak scratch per call is
now bounded at chunk*K*2 bytes instead of N*K*2 (52-88 MB per call for the big
layers).

A/B interleaved best-of-3 on the same (loaded) box, K=N=5120, threads=8 —
both binaries built back-to-back so contention affects both equally:

| M | before (full widen) | after (chunk 64 / no widen) | delta |
|---|---|---|---|
| 64 | 48.9 ms (68.7 TOPS) | **16.5 ms (203.5 TOPS)** | **3.0x** |
| 512 | 149.8 ms | 142.7 ms | ~1.05x |
| 2048 | 591.6 ms | 569.9 ms | ~1.04x |
| 16384 | 4744 ms | 4629 ms | ~1.02x |

Quiet-box spot check of the after-code: M=64 15.0-18.5 ms (181-224 TOPS),
M=16384 4391 ms. M=64 is the shape the whole 64-token forward uses, so the 3x
kernel win lands ~500 times per forward. `test_core` qgemm cosines unchanged
(0.999992 at all shapes).

### 2. Audio ConvMLP patch embed via im2col + matmul (`src/model/ovi_dit.cpp`)

The scalar `conv1d_pad` (stride-7 gather over [8192,3072,7] weights, ~176 s of
the 176.4 s "other" bucket) is replaced by `conv1d_mm`: im2col [L, Cin*7] ->
one `sd::matmul` per conv (OpenBLAS at these shapes, ~265 GFLOPS). Conv weights
are now stored TRANSPOSED [Cin*7, Cout] once at load — for int8 packs the
dequant goes straight into the transposed layout (no extra ~700 MB x 3 copy;
pack scales are per-(out,in) 7-tap vectors, `sdcpp` last-dim quantization).
Forward-section effect (warm, 64 tokens): **"other" 176.4 s -> 2.7-3.6 s**
(ConvMLP itself ~176 s -> ~1 s: 26 GFLOP of conv + 6.3 GB of now-sequential
weight reads). Parity: `test_ovit` out_audio rel-err 8.7e-7 (gate 1e-3),
`test_ovit8` int8-vs-fp32 out_audio cosine 0.9998 (unchanged from baseline).

### 3. sdpa restructure (`src/model/blocks.cpp`)

Old: per-(head,row) tasks (grain 8 => ~192 tiny tasks/call), scalar dot loops,
strided k/v gathers, per-task vector alloc. New: K transposed once per call
into [H*hd, Lk] (contiguous), then one `parallel_for` over heads; per head a
tiled AVX2 QK^T over 8-wide j-tiles, softmax over one contiguous S row, fused
AV with 8 accumulators over head_dim. Same math (no flash tiling/online
softmax — exact), one task-local scratch vector per head.
// ponytail: no struct workspace — parallel_for spawns fresh threads per call,
so a SelfAttn-member buffer would need locking; per-head task-local is cheaper.
Warm sdpa section, 64 tokens: **3.1 s -> 1.7 s** (~1.8x, under load).
`test_blocks` self_attn/cross_attn rel-err ~1e-6 (gate 1e-3).

### End-to-end (64 tokens, real 12 GB pack)

After all three (test_ovit8 part 2, `make` build, warm best-of-3):

| phase | before | after |
|---|---|---|
| forward warm | 205.2 s (quiet box) | **19.0-19.6 s** (4 passing runs, box load 5-12 from parallel agents) |
| — lin (qgemm) | 25.7 s | 14.4-14.6 s |
| — sdpa | 3.1 s | 1.7 s |
| — other | 176.4 s | 2.9-3.3 s |
| int8 parity | 0.9999/0.9998 | 0.9999/0.9998 (unchanged) |

The after-numbers were taken while 1-2 other agents kept the 8-core box at
load 5-12 (T5 prompts in the same runs were inflated up to 200x vs quiet), so
~19.5 s is an upper bound; the section floors below suggest ~12-14 s on a
quiet box — under the 15 s target, not the 5 s stretch.

Remaining bottleneck, in order:
- **lin 14.4 s** — arithmetic-bound at M=64: ~11 G int8 weights x ~70 avg
  rows = ~1.5 TMAC per forward; at the measured M=64 rate (181-224 TOPS) that
  is a **~7.5-8.5 s kernel floor**. (Streaming floor is lower: warm page-cache
  read of the pack measures ~4.7 GB/s single-thread, ~2.5 s for 11 GB serial,
  and the dots already overlap it.) To cut it: batch CFG's 2 forwards (M=128)
  or denoise steps' shared text ctx, or a VNNI dot — this CPU has no
  AVX-VNNI/AVX512 (cpuinfo), so the madd path is the per-core ceiling here.
- **other 2.9 s** — ConvMLP ~1 s (OpenBLAS) + per-layer elementwise/mod ops in
  `ovi_dit.cpp` that are still serial loops (mod_rows, gated_add_inplace,
  add_inplace) and per-call Tensor allocs; parallelizing them is the next
  cheap win at ~15% of runtime.
- **sdpa 1.7 s** — now compute-shaped; at 64-192 tokens it stays ~9% of
  runtime. Quadratic term only matters at multi-k tokens.

### 192-token bench (SD_BENCH_TOKENS=192 -> latent [48,3,16,16], ~256x256 x 3 frames)

Same run conditions (load 9-13): warm best-of-3 **40.8 s/step** =
lin 30.2 + sdpa 4.2 + other 6.3. Scaling vs 64 tokens: lin 2.1x (weights
stream once regardless; arithmetic per weight-byte tripled), other 2.2x
(~linear in tokens), sdpa 2.4x (cross-attn linear + Lk=512 constant dominates
over the small quadratic part). 4-step CFG micro-generation = 8 forwards:
**64 tok: 156.6 s measured; 192 tok: 326 s measured under load, ~3-3.5 min
quiet-box estimate**.

## Token-scaling projection to 720p 5s (part 3)

Formula (exact token math from `ovi_dit.cpp`): `tokens = F * floor(H/2) * floor(W/2)`;
real latent [48,31,45,45], patch (1,2,2) -> `tokens_real = 31*22*22 = 15004`
(padded-to-even would be 31*23*23 = 16399). `r = 15004/64 = 234.4`.

```
projected_step = lin_ms * r            (gemms ~linear in tokens; upper bound at
                                        64 tok, which is weight-streaming-bound)
               + attn_ms * r^2         (video self/fusion sdpa is quadratic;
                                        cross-attn is linear -> upper bound)
= 25664*234.4 + 3134*234.4^2 ms
= 6'016'649 + 172'244'584 ms  ≈ 49.5 h/step
50-step clip ≈ 148'551 min ≈ 103 days (x2 with CFG)
```

FLOP cross-check (sdpa at 64 tokens is overhead-dominated, so r^2 overestimates):
video self+fusion attention at 15k tokens ≈ 165 TFLOP fp32/forward; at the
measured 26–265 GFLOPS kernel rates that is 10 min–1.8 h per step of attention
alone; qgemm linears ≈ 190 TFLOP int8 ≈ 15 min/step at 220 TOPS. Either way a
full-quality 720p 5s clip is **days-to-months on this box** vs 83 s on one H100
(~10^4–10^5x gap) — confirming the README's reduced-token strategy: lower
resolution/frame counts cut tokens linearly (linears) and quadratically
(attention), which is the only lever that makes local generation finish in
minutes. With the #17 perf pass landed (205 s -> 19 s @64 tok, 41 s @192 tok),
the same projection from the measured 192-token point is ~27.9 s x r(=78.1)
lin + quadratic attn ≈ **~7.8 h/step, ~16 days for a 50-step CFG 720p 5s clip**
— still H100-territory or bust. Next levers, in order of projected win:
batch CFG pairs into one qgemm (2x M amortization for free), parallelize the
serial elementwise loops in `ovi_dit.cpp` (~15% at low tokens), blocked/flash
sdpa (only matters >=1k tokens), fused dequant+gemm (kills the remaining i8
round-trips at high M).

## Verification (issue #17)

`test_core` (matmul/qgemm/quantize cosines), `test_blocks` (block golden
parity <1e-3), `test_ovit` (fp32 golden parity 4.6e-7/8.7e-7 + key coverage),
`test_t5enc` (bf16 golden parity, cache-off default), `test_ovit8` (int8
parity cosine 0.9999/0.9998 > 0.99, T5 cache golden parity, real bench +
projection at 64 tokens and SD_BENCH_TOKENS=192) — all pass after the perf
pass. Bench caveat: this box is shared with other agents; the 19.0-19.4 s
warm 64-token number and the 40.8 s 192-token number were taken under load
5-13 (baseline 205.2 s was quiet-box, so the >=10.8x speedup is a lower
bound).
