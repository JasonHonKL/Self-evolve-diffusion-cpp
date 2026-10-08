// UMT5-XXL text encoder (Wan2.2 T5), ported from Ovi/ovi/modules/t5.py.
// Architecture (umt5_xxl, encoder_only): vocab 256384, dim=dim_attn=4096,
// dim_ffn=10240, num_heads=64 (head_dim 64), 24 encoder layers, PER-LAYER
// relative position bias (shared_pos=False, num_buckets=32, bidirectional,
// max_dist=128), gated-GELU FFN, RMSNorm (eps 1e-6, no bias/mean), attention
// WITHOUT 1/sqrt(d) scaling, softmax in fp32, pad mask -> -inf on keys.
//
// Weights: bf16 views into the mmap'd safetensors from tools/convert_t5.py,
// which stores Linear weights already transposed to [in, out] (torch W^T).
// sd::matmul converts bf16 -> f32 per call (gemm.cpp:119-120) — correct but
// re-converts ~780 MB/layer on every forward; fine for tests, cache fp32
// weights if it ever becomes the bottleneck.
#pragma once
#include "serde/safetensors.h"
#include <string>
#include <vector>

namespace sd {
namespace t5 {

struct T5Encoder {
  static constexpr int dim = 4096;
  static constexpr int dim_ffn = 10240;
  static constexpr int num_heads = 64;
  static constexpr int num_layers = 24;
  static constexpr int num_buckets = 32;
  static constexpr int head_dim = dim / num_heads;
  static constexpr int vocab = 256384;

  SafetensorsFile st;  // owns the mmap; weight tensors are bf16 views into it
  std::vector<Tensor> norm1_w, norm2_w, pos_w;  // small per-layer fp32 copies
  Tensor final_norm_w;

  explicit T5Encoder(const std::string& safetensors_path);

  // ids: F32 [L] (values cast to int), mask: F32 [L] (nonzero = keep token).
  // out: F32 [L, dim] final hidden states (after last RMSNorm).
  void forward(const Tensor& ids, const Tensor& mask, Tensor& out) const;
};

}  // namespace t5
}  // namespace sd
