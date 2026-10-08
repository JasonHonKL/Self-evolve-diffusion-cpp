# Self-evolve-diffusion-cpp

A from-scratch, CPU-first C++17 inference framework for diffusion video models —
built in the spirit of [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),
targeting **Ovi** (Character.AI's 12B twin-backbone audio+video DiT, Wan2.2-TI2V-5B
video branch + 5B audio branch) and small enough to run anywhere.

No CUDA, no cloud, no PyTorch at inference time. Own kernels, mmap'd weights,
int8 weight quantization, POSIX threads, OpenBLAS when available.

**Why C++ and not Rust:** the reference codebase (sd.cpp, GGML) is C/C++,
OpenBLAS exposes a C ABI, and the repo is named `-cpp`. Same performance class,
faster interop with the ecosystems we port from. (A Rust rewrite of the kernel
layer is a non-goal until the C++ layer is complete — see issue ledger.)

## Target hardware reality (this repo was born on it)

- 8 CPU cores, 23 GB RAM, **no GPU**, measured **0.16 TFLOPS** fp32 (OpenBLAS)
- SOTA reference: Ovi 5s/720p/50 steps = **83 s on one H100** (~990 TFLOPS)
- Implication: full-quality local generation is compute-bound by ~6000x.
  This framework chases the honest optimum: **int8 + every FLOP accounted for**,
  plus reduced-token modes (small resolution / frame counts) that make a
  *real local generation* finish in minutes, not days.

## Architecture

```
src/
  core/        tensor, GEMM (fp32/bf16 SIMD + threaded), int8 qGEMM, thread pool
  serde/       safetensors mmap loader, .sdcpp packed weight format
  tokenizer/   UMT5 sentencepiece (unigram Viterbi) tokenizer
  model/       DiT block zoo: RMSNorm, RoPE-3D, SDPA, AdaLN, FFN, cross-attn,
               dual-backbone fusion; UMT5-XXL encoder; Wan2.2 causal 3D VAE
  sampler/     UniPC multistep solver (ported from Ovi's fm_solvers_unipc.py)
  pipeline/    denoise loop, CLI (movigen), movie driver
tools/
  quantize/    bf16 safetensors -> int8 .sdcpp packer
  golden/      numpy golden-vector generators for C++ parity tests
tests/         unit + parity tests, run via `make test`
```

## Build

```
make          # -O3 -march=native, links OpenBLAS if found
make test     # unit + parity
make bench    # kernel benchmarks (TFLOPS achieved vs ceiling)
```

## Milestones

- **M0 Foundation** — build, tensor core, GEMM, int8 qGEMM, safetensors,
  tokenizer, sampler, block zoo, golden tests *(in progress)*
- **M1 Forward passes** — UMT5 encoder, DiT block parity, full 28-layer fusion DiT
- **M2 Decode** — Wan2.2 3D VAE decoder, MMAudio vocoder, end-to-end denoise loop
- **M3 End to end** — weight packer, `movigen` CLI, short-movie scene driver

See [DEPENDENCIES.md](DEPENDENCIES.md) for the issue/dependency graph.

## License

MIT
