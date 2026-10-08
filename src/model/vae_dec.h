// Wan2.2-TI2V-5B causal 3D VAE decoder, ported from Ovi/ovi/modules/vae2_2.py
// (Wan2_2_VAE -> WanVAE_(dec_dim=256) -> Decoder3d). Encoder not needed.
//
// Arch (verified against source + checkpoint, tools/convert_vae.py output):
//   z_dim=48, decoder base dim 256, dim_mult [1,2,4,4] -> stage dims
//   [1024,1024,1024,512,256]; 3 resblocks/stage (num_res_blocks+1); middle =
//   2x ResBlock(1024) + single-head AttentionBlock(1024); temperal_upsample
//   [T,T,F]: stages 0,1 Resample(upsample3d) = causal time_conv(1024->2048,
//   k(3,1,1)) doubling + nearest-2x + Conv2d 3x3; stage 2 upsample2d; stage 3
//   none. Each up stage adds a DupUp3D nearest-up shortcut (Up_ResidualBlock).
//   Head: RMS_norm -> SiLU -> CausalConv3d(256->12) -> unpatchify(2) -> RGB.
//   All convs groups=1. Compression: 16x spatial (8x conv + 2x unpatchify),
//   temporal 4x CAUSAL: latent [48,F,H,W] -> [3, 4F-3, 16H, 16W] (first latent
//   frame is the anchor -> 1 rgb frame; each later one -> 4 frames).
//
// Streaming semantics mirror the reference feat_cache protocol exactly: every
// causal k3 site keeps a 2-frame history (zeros at stream start; the time_conv
// sites additionally skip the anchor frame). Whole-tensor decode() therefore
// equals any chunking of decode_chunk() up to GEMM accumulation order.
//
// RAM estimates (fp32 activations, internal layout [T,C,H,W]):
//   whole decode of a 720p latent [48,81,90,160] peaks >200 GB (stage-2/3
//   tensors are [4F,512,1280,720]) — do not do that. Chunked (4 latent
//   frames): biggest live tensors ~[16,512,1280,720] + [16,1024,640,360] +
//   im2col tiles (~256 MB) + fp32 weights 2.2 GB -> peak ~12-15 GB.
//   Measured: whole [48,8,32,32] (29 out frames @512x512) peaks 19.8 GB RSS
//   and takes ~19 min on 8 cores (im2col+sgemm ~130 TFLOP); same latent
//   chunked (4-frame chunks) roughly halves the activation peak.
#pragma once
#include "serde/safetensors.h"
#include <string>

namespace sd {
namespace vae {

struct VaeDecoder {
  static constexpr int z_dim = 48;

  // Causal conv (kt 1 or 3) with optional preceding RMS_norm gamma.
  // w: f32 [Cin*kt*kd*kd, Cout] matmul-ready (torch [Cout,Cin,kt,kd,kd]
  // permuted at load). kd = spatial kernel (1 or 3).
  struct Conv {
    Tensor w, b, gamma;
    int Cin = 0, Cout = 0, kt = 1, kd = 1;
  };

  Conv conv2;        // 48->48 1x1x1, applied to scaled z (stateless)
  Conv conv1;        // 48->1024 k3 causal
  Conv mid[4];       // middle.0/.2 ResidualBlocks x 2 convs, 1024
  struct {
    Tensor gamma, qkv_w, qkv_b, proj_w, proj_b;  // qkv_w [1024,3072] etc
  } attn;
  Conv up[4][3][2];  // [stage][resblock][conv]; sc only for stage2/3 rb0
  Conv up_sc[2];     // stage2 rb0 (1024->512), stage3 rb0 (512->256) 1x1x1
  Conv tconv[2];     // stage0/1 Resample.time_conv 1024->2048 (3,1,1)
  Conv sconv[3];     // stage0/1/2 Resample Conv2d 3x3 (1024/1024/512 ch)
  Conv head;         // 256->12 k3 causal (gamma = head RMS_norm)
  Tensor mean_, std_;  // [48] f32; zs = z*std + mean before conv2

  SafetensorsFile st;  // mmap owner; bf16 views converted to f32 at load

  // Per-site causal history: last 2 input frames [C,H,W] (zeros at start).
  struct Hist {
    Tensor f[2];
    bool started = false;
  };
  struct Cache {
    Hist h[32];  // 30 resblock/entry/head sites + 2 time_conv sites
    bool anchor_done = false;
  };

  explicit VaeDecoder(const std::string& safetensors_path);

  // z: F32 [48,F,H,W] raw latent (torch order). out: F32 [3,4F-3,16H,16W].
  // Streams anchor frame + one big chunk through the cache machinery.
  void decode(const Tensor& z, Tensor& out);

  // Streaming decode of one chunk of latent frames [48,T,H,W]; the FIRST
  // chunk of a stream must contain latent frame 0 (the anchor). Emits this
  // chunk's frames: [3, 4T-3] for the anchor chunk, [3, 4T] afterwards.
  void decode_chunk(const Tensor& z, Tensor& out, Cache& cache);
};

}  // namespace vae
}  // namespace sd
