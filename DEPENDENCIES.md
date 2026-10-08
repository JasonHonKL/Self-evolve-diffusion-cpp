# Issue / dependency graph

Mermaid: `A --> B` means "A is a dependency of B".

```mermaid
graph TD
    T1[#1 toolchain+build] --> T2
    T2[#2 tensor core + fp32 GEMM] --> T3
    T2 --> T5
    T3[#3 int8 qGEMM] --> T15
    T4[#4 safetensors mmap loader] --> T9
    T4 --> T10
    T4 --> T15
    T5[#5 golden parity harness] --> T10
    T5 --> T6
    T6[#6 UniPC sampler port] --> T14
    T7[#7 UMT5 tokenizer] --> T9
    T8[#8 DiT block zoo] --> T9
    T8 --> T10
    T9[#9 UMT5-XXL encoder fwd] --> T16
    T10[#10 DiT block fwd parity] --> T11
    T11[#11 full fusion DiT fwd] --> T14
    T12[#12 Wan2.2 3D VAE decoder] --> T16
    T13[#13 MMAudio vocoder+mux] --> T16
    T14[#14 end-to-end denoise loop] --> T16
    T15[#15 weight quantizer/packer] --> T16
    T16[#16 movigen CLI t2v] --> T18
    T17[#17 bench+opt ledger] --> T18
    T18[#18 short-movie scene driver]
```

| # | Issue | Milestone | Depends on |
|---|-------|-----------|------------|
| 1 | Toolchain + Makefile + CI-green `make test` | M0 | — |
| 2 | Tensor core: Tensor, blocked+SIMD threaded fp32 GEMM, bf16↔fp32 | M0 | 1 |
| 3 | Int8 qGEMM (s8s8→s32, per-channel scales) + accuracy test | M0 | 2 |
| 4 | Safetensors mmap loader (bf16/f32/f8) | M0 | 1 |
| 5 | Golden parity harness (numpy ref → npz → C++ tests) | M0 | 2 |
| 6 | UniPC sampler port from `Ovi/ovi/utils/fm_solvers_unipc.py` | M0 | 5 |
| 7 | UMT5 sentencepiece unigram tokenizer | M0 | 1 |
| 8 | DiT block zoo: RMSNorm, RoPE-3D, SDPA, AdaLN, FFN, cross-attn | M0 | 2 |
| 9 | UMT5-XXL encoder forward pass | M1 | 4, 7, 8 |
| 10 | DiT block forward parity vs golden | M1 | 4, 5, 8 |
| 11 | Full 28-layer dual-backbone fusion DiT forward | M1 | 10 |
| 12 | Wan2.2 causal 3D VAE decoder | M2 | 2, 4, 8 |
| 13 | MMAudio vocoder + audio mux | M2 | 2, 4, 8 |
| 14 | UniPC end-to-end denoise loop | M2 | 6, 11 |
| 15 | Weight quantizer: bf16 safetensors → int8 .sdcpp pack | M2 | 3, 4 |
| 16 | `movigen` CLI: text → mp4 (with audio) | M3 | 9, 11, 12, 13, 14, 15 |
| 17 | Benchmarks + optimization ledger (TFLOPS vs 0.16 ceiling) | M3 | 2, 3 |
| 18 | Short-movie driver: scene script → stitched clips | M3 | 16, 17 |

Session-1 target: close #1–#8, file #9–#18 with detailed specs.
