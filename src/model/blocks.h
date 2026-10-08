// DiT layer zoo for Ovi's fusion transformer, ported from Ovi/ovi/modules/model.py
// (+ fusion.py wiring). All math cites the Python source. fp32 throughout.
#pragma once
#include "core/tensor.h"

namespace sd {
namespace blk {

// Dims from Ovi/ovi/configs/model/dit/video.json and audio.json (identical):
// dim=3072, ffn_dim=14336, num_heads=24, eps=1e-6, qk_norm, cross_attn_norm.
// Audio-only extra: temporal_rope_scaling_factor=0.19676 (see rope_freqs_1d).
struct Config {
  static constexpr int dim = 3072;
  static constexpr int ffn_dim = 14336;
  static constexpr int num_heads = 24;
  static constexpr int head_dim = dim / num_heads;  // 128
  static constexpr float eps = 1e-6f;
  static constexpr float rope_theta = 10000.0f;  // model.py:38 default
  static constexpr float audio_rope_scaling = 0.19676f;  // audio.json
};

// Row-major matmul-ready linear: y = x @ w + b, w shape [in, out] (= torch W^T).
struct Linear {
  Tensor w;
  Tensor b;
  void init(int64_t in, int64_t out) {
    w = Tensor({in, out}, DType::F32);
    b = Tensor({out}, DType::F32);
  }
};

void linear(const Tensor& x, const Linear& l, Tensor& y);

// LayerNorm over last dim; w/b null => non-affine (model.py:185-195 WanLayerNorm,
// eps=1e-6; norm1/norm2 non-affine, norm3 affine when cross_attn_norm=true).
void layernorm(const Tensor& x, Tensor& y, const Tensor* w, const Tensor* b,
               float eps);

// ---- RoPE3D / RoPE1D (model.py:38-45 rope_params, 669-686 set_rope_params,
// 72-100 rope_apply_3d/rope_apply_1d). theta=10000; per head of head_dim=128:
// 64 complex pairs split t=22 / h=21 / w=21 (real dims 44/42/42:
// t_real = d - 4*(d//6), s_real = 2*(d//6), freq_j = theta^(-2j/real)).
// 3D rotates all 64 pairs; 1D (audio) rotates only the first 22, freqs scaled
// by temporal_rope_scaling_factor.
struct RopeFreqs {
  Tensor ang;  // [L, n_rot] rotation angle (radians) per token per complex pair
  int n_rot = 0;
};

RopeFreqs rope_freqs_3d(int T, int H, int W,
                        int head_dim = Config::head_dim,
                        float theta = Config::rope_theta);
RopeFreqs rope_freqs_1d(int L, int head_dim = Config::head_dim,
                        float theta = Config::rope_theta,
                        float scaling = 1.0f);

// In-place on x [L, C]; pair j of each head rotated by ang[l][j], j < n_rot.
void apply_rope(Tensor& x, const RopeFreqs& f,
                int num_heads = Config::num_heads);

// Scaled dot-product attention, scale = head_dim^-0.5 (flash_attention default,
// attention.py:25-52). q [Lq,C], k/v [Lk,C] -> out [Lq,C]. `causal` unused by
// the source (no first-frame mask exists; first_frame_is_clean only zeroes the
// timestep, model.py:734-742) but kept per issue spec.
void sdpa(const Tensor& q, const Tensor& k, const Tensor& v, int num_heads,
          Tensor& out, bool causal = false);

// Self-attention (model.py:198-265). q,k,v linears fused into one [dim,3*dim]
// weight (source has 3 separate biased Linears:216-218 — concat-equivalent).
// qk RMSNorm over the FULL dim (eps=1e-6, weighted), RoPE on q,k only.
struct SelfAttn {
  Linear qkv, o;
  Tensor nq_w, nk_w;  // RMSNorm weights [dim], model.py:220-221
  float eps = Config::eps;

  void init(int64_t dim = Config::dim) {
    qkv.init(dim, 3 * dim);
    o.init(dim, dim);
    nq_w = Tensor({dim}, DType::F32);
    nk_w = Tensor({dim}, DType::F32);
  }
  void forward(const Tensor& x, const RopeFreqs& fr, Tensor& out) const;
};

// Cross-attention, t2v type (model.py:268-294 — both video 'ti2v' and audio
// 't2a' map to WanT2VCrossAttention, model.py:642). q from x, k/v from ctx,
// qk RMSNorm on q and k, NO rope, no modulation branch.
struct CrossAttn {
  Linear q, k, v, o;
  Tensor nq_w, nk_w;
  float eps = Config::eps;

  void init(int64_t dim = Config::dim) {
    q.init(dim, dim);
    k.init(dim, dim);
    v.init(dim, dim);
    o.init(dim, dim);
    nq_w = Tensor({dim}, DType::F32);
    nk_w = Tensor({dim}, DType::F32);
  }
  void forward(const Tensor& x, const Tensor& ctx, Tensor& out) const;
};

// FFN (model.py:420-422): Linear(dim,ffn_dim) -> GELU(tanh approx) ->
// Linear(ffn_dim,dim). Biased. NOT gated (wan2.1-style, not 2.2 gated-silu).
struct FFN {
  Linear fc1, fc2;
  void init(int64_t dim = Config::dim, int64_t ffn_dim = Config::ffn_dim) {
    fc1.init(dim, ffn_dim);
    fc2.init(ffn_dim, dim);
  }
  void forward(const Tensor& x, Tensor& y) const;
};

// Modulation apply convention (model.py:454-467): chunk order
// [shift1,scale1,gate1,shift2,scale2,gate2]; h = ln(x)*(1+scale)+shift;
// residual x + h*gate.
void mod_shift_scale(const Tensor& ln_x, const Tensor& shift,
                     const Tensor& scale, Tensor& y);

// One composite transformer block = WanAttentionBlock.forward (model.py:430-471):
// x += gate1 * self_attn(ln1(x)*(1+e1)+e0)
// x += cross_attn(ln3_affine(x))                       [no modulation]
// x += gate2 * ffn(ln2(x)*(1+e4)+e3)
// e [L,6,dim] gets ModulationAdd bias [6,dim] added first (model.py:368-374).
struct TransformerBlock {
  Tensor ln3_w, ln3_b;  // affine LN (cross_attn_norm=true, model.py:401-403)
  SelfAttn self_attn;
  CrossAttn cross_attn;
  FFN ffn;
  Tensor mod_bias;  // [6, dim]
  float eps = Config::eps;

  void init(int64_t dim = Config::dim, int64_t ffn_dim = Config::ffn_dim) {
    self_attn.init(dim);
    cross_attn.init(dim);
    ffn.init(dim, ffn_dim);
    ln3_w = Tensor({dim}, DType::F32);
    ln3_b = Tensor({dim}, DType::F32);
    mod_bias = Tensor({6, dim}, DType::F32);
  }
  void forward(const Tensor& x, const Tensor& e, const RopeFreqs& fr,
               const Tensor& ctx, Tensor& out) const;
};

}  // namespace blk
}  // namespace sd
