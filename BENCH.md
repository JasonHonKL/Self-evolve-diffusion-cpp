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
minutes. Next levers, in order of projected win: blocked/flash sdpa, a matmul
form of the audio ConvMLP, fused dequant+gemm to kill the widen pass.

## Verification (issue #17)

`test_ovit` (fp32 golden parity 1e-6 + key coverage), `test_t5enc` (bf16 golden
parity, cache-off default), `test_ovit8` (int8 parity cosine > 0.99, T5 cache
golden parity, real bench + projection) — all pass.
