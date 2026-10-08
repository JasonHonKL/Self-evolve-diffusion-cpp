# MMAudio vocoder port (issue #13)

C++17 inference for Ovi's full audio path: **DiT audio latent → WAV**.
Ported, not descoped. Parity vs torch fp32 golden: **cosine 1.000 (mel), 0.9999 (waveform)**.

## The actual network chain (verified from Ovi source)

Ovi does NOT vocode at 44.1kHz. `init_mmaudio_vae` uses `mode='16k'`
(`Ovi/ovi/utils/model_loading_utils.py`), so the chain is:

```
DiT audio latent [1, 20, T] @ 31.25 Hz   (T=157 for 5s, T=314 for 10s;
                                           in_dim=out_dim=20 in audio.json)
  │  MMAudio VAE decoder (ext_weights/v1-16.pth, 99.1M params)
  ▼  mel: 80-band log10-mel @ 62.5 Hz, hop 256, fmax 8k   [1, 80, 2T]
  │  BigVGAN v1 generator (ext_weights/best_netG.pt, 112.2M params)
  ▼  waveform @ 16 kHz, tanh-limited                      [1, 1, 512T]
```
(`wrapped_decode` = VAE decode then `vocode`; Ovi runs it in bf16 autocast —
this port is fp32, which is the *tighter* parity target.)

### Stage 1: VAE Decoder1D (EDM2-style, all Conv1d k=3/k=1, no biases)
- `conv_in` MPConv 20→1536; mid: 2 ResnetBlocks + 1 self-attention block
  (1536 ch, single head, qkv produced by one k=1 conv with **q/k/v
  interleaved per channel** — channel `3c+w`; per-C pixel-norm on q,k,v;
  softmax scale 1/sqrt(1536))
- ResnetBlock: pixel-norm (x / (1e-4 + ||x||·C^-1/2)) → mp_silu → conv →
  mp_silu → conv → mp_sum((0.7a+0.3b)/sqrt(0.58), a=shortcut); clamp ±256
  after every block
- up levels 2→1→0 (1536→768→384, 3 blocks each); after level 1: nearest ×2
  + MPConv
- `conv_out` 384→80 scaled by learnable gain + 1, then unnormalize with the
  80-d DATA_MEAN/STD constants (baked into v1-16.pth buffers)
- MPConv weights are magnitude-preserving-normalized at conversion time
  (`remove_weight_norm`), so C++ sees plain convs

### Stage 2: BigVGAN v1 generator (best_netG.pt → bigvgan_vocoder.yml)
- conv_pre Conv1d 80→1536 k7
- 6 upsampling stages: ConvTranspose1d k=(8,8,4,4,4,4), stride=(4,4,2,2,2,2)
  (=256× total = mel hop), channels 1536→768→384→192→96→48→24; weight layout
  **[in, out, k]** (ConvTranspose1d), permuted at load
- per stage 3 AMPBlocks (k=3,7,11; dilations 1,3,5): 3× [act → dilated conv →
  act → conv d=1 → residual]; outputs averaged
- activation = alias-free SnakeBeta: kaiser-sinc upsample ×2 (kernel 12,
  replicate pad) → `x + sin²(αx)/β` (α, β per channel, logscale → exported
  pre-exp) → kaiser-sinc downsample ×2
- activation_post (24ch) → conv_post 24→1 k7 → tanh
- weight_norm (g,v) folded at conversion; kaiser filters are fixed buffers,
  exported as flat vectors

## Files

| file | role |
|---|---|
| `tools/convert_vocoder.py` | both .pth → `ckpts/mmaudio_vocoder.safetensors` (845MB, folds weight_norm/MPConv norm, exp's snake params) |
| `src/model/vocoder.h/.cpp` | conv1d (im2col+GEMM, tiled), ConvTranspose1d (tap-column GEMM), alias-free snake, VAE decoder, BigVGAN; `set_trace()` debug hook |
| `tools/golden/gen_voc_golden.py` | torch fp32 forward + 24 stage snapshots → `tests/golden/voc_golden.npz` |
| `tests/test_vocoder.cpp` | parity (cos>0.98) + timing at T=157/314 |
| `src/pipeline/mux.h/.cpp` | f32 WAV writer + ffmpeg mux (`-c:v copy -c:a aac -shortest`) |
| `tests/test_mux.cpp` | 1s sine → mux onto ffmpeg-made black mp4 → ffprobe asserts audio stream |

## Run

```
python3 tools/convert_vocoder.py && python3 tools/golden/gen_voc_golden.py
make build/tests/test_vocoder build/tests/test_mux
./build/tests/test_vocoder && ./build/tests/test_mux
```

## Results (8-core EPYC, OpenBLAS)

```
mel  [80,128]: cosine=1.000000  PASS
wav  [1,32768]: cosine=0.999884  PASS
full chain:     cosine=0.999884  PASS
T=157 (5s):  vae 1.1s + bigvgan 29.6s = 30.7s  (0.16x realtime)
T=314 (10s): vae 1.0s + bigvgan 54.7s = 55.7s  (0.18x realtime)
```

Bugs found while porting (all fixed, worth remembering):
1. `sd::matmul`'s `bias` argument is per-**column** (Linear convention);
   conv bias is per-channel = per-**row** in channels-first [C,T] GEMMs.
2. Tiled conv im2col must offset tile validity bounds by the tile start
   (t0), not treat each tile as the sequence start.
3. `mp_sum(x, h)` argument order matters (0.7/0.3 asymmetric weighting).

`make test` note: `test_ovit8` OOMs on this machine (21.8GB peak vs 23GB
RAM, 12GB `ckpts/Ovi/model.sdcpp` load) — reproduces with this repo's
vocoder/mux objects excluded; pre-existing, unrelated to this port.
