// Implementation for src/model/blocks.h. Built only on sd:: core + STL.
#include "model/blocks.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sd {
namespace blk {

void linear(const Tensor& x, const Linear& l, Tensor& y) {
  matmul(x, l.w, y, l.b.data ? l.b.ptr<float>() : nullptr);
}

void layernorm(const Tensor& x, Tensor& y, const Tensor* w, const Tensor* b,
               float eps) {
  y = Tensor(x.shape, DType::F32);
  int64_t M = x.rows(), N = x.cols();
  const float* xp = x.ptr<float>();
  float* yp = y.ptr<float>();
  const float* wp = w ? w->ptr<float>() : nullptr;
  const float* bp = b ? b->ptr<float>() : nullptr;
  parallel_for(0, M, 1, [&](int64_t bb, int64_t ee) {
    for (int64_t i = bb; i < ee; i++) {
      const float* r = xp + i * N;
      float mu = 0;
      for (int64_t k = 0; k < N; k++) mu += r[k];
      mu /= (float)N;
      float var = 0;
      for (int64_t k = 0; k < N; k++) {
        float d = r[k] - mu;
        var += d * d;
      }
      float inv = 1.f / std::sqrt(var / (float)N + eps);
      float* o = yp + i * N;
      for (int64_t k = 0; k < N; k++)
        o[k] = (r[k] - mu) * inv * (wp ? wp[k] : 1.f) + (bp ? bp[k] : 0.f);
    }
  });
}

// model.py:38-45,669-686: t_real = d - 4*(d//6), s_real = 2*(d//6),
// freq_j = theta^(-2j/real); 3D axes (t,h,w) concatenated in that order.
RopeFreqs rope_freqs_3d(int T, int H, int W, int head_dim, float theta) {
  int d = head_dim;
  int t_real = d - 4 * (d / 6), s_real = 2 * (d / 6);
  int ct = t_real / 2, cs = s_real / 2;
  RopeFreqs f;
  f.n_rot = ct + 2 * cs;
  f.ang = Tensor({(int64_t)T * H * W, f.n_rot}, DType::F32);
  std::vector<float> ft(ct), fs(cs);
  for (int j = 0; j < ct; j++) ft[j] = std::pow(theta, -2.f * j / t_real);
  for (int j = 0; j < cs; j++) fs[j] = std::pow(theta, -2.f * j / s_real);
  float* a = f.ang.ptr<float>();
  for (int t = 0; t < T; t++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++) {
        float* r = a + ((int64_t)t * H + h) * W * f.n_rot + w * f.n_rot;
        int j = 0;
        for (; j < ct; j++) r[j] = t * ft[j];
        for (; j < ct + cs; j++) r[j] = h * fs[j - ct];
        for (; j < f.n_rot; j++) r[j] = w * fs[j - ct - cs];
      }
  return f;
}

// Audio 1D rope (model.py:679): only the t-axis part, freqs scaled.
RopeFreqs rope_freqs_1d(int L, int head_dim, float theta, float scaling) {
  int d = head_dim;
  int t_real = d - 4 * (d / 6);
  RopeFreqs f;
  f.n_rot = t_real / 2;
  f.ang = Tensor({L, f.n_rot}, DType::F32);
  float* a = f.ang.ptr<float>();
  for (int l = 0; l < L; l++)
    for (int j = 0; j < f.n_rot; j++)
      a[(int64_t)l * f.n_rot + j] =
          l * scaling * std::pow(theta, -2.f * j / t_real);
  return f;
}

// model.py:72-100: complex pair j of each head multiplied by e^{i*ang[l][j]}.
void apply_rope(Tensor& x, const RopeFreqs& f, int num_heads) {
  int64_t L = x.rows(), C = x.cols(), hd = C / num_heads;
  float* p = x.ptr<float>();
  const float* a = f.ang.ptr<float>();
  int nr = f.n_rot;
  parallel_for(0, L, 1, [&](int64_t bb, int64_t ee) {
    for (int64_t l = bb; l < ee; l++)
      for (int h = 0; h < num_heads; h++)
        for (int j = 0; j < nr; j++) {
          float c = std::cos(a[l * nr + j]), s = std::sin(a[l * nr + j]);
          float* q = p + l * C + h * hd + 2 * j;
          float r0 = q[0], r1 = q[1];
          q[0] = r0 * c - r1 * s;
          q[1] = r0 * s + r1 * c;
        }
  });
}

// ponytail: O(Lq*Lk*hd) per head, no flash tiling; fine at test sizes, revisit
// with blocked softmax if real seq lens (>=36k tokens) ever run on this path.
void sdpa(const Tensor& q, const Tensor& k, const Tensor& v, int num_heads,
          Tensor& out, bool causal) {
  int64_t Lq = q.rows(), Lk = k.rows(), C = q.cols(), hd = C / num_heads;
  float scale = 1.f / std::sqrt((float)hd);
  out = Tensor({Lq, C}, DType::F32);
  const float* qp = q.ptr<float>();
  const float* kp = k.ptr<float>();
  const float* vp = v.ptr<float>();
  float* op = out.ptr<float>();
  parallel_for(0, (int64_t)num_heads * Lq, 8, [&](int64_t bb, int64_t ee) {
    std::vector<float> sc(Lk);
    for (int64_t idx = bb; idx < ee; idx++) {
      int h = (int)(idx / Lq), i = (int)(idx % Lq);
      const float* qi = qp + (int64_t)i * C + h * hd;
      float mx = -INFINITY;
      for (int64_t j = 0; j < Lk; j++) {
        const float* kj = kp + j * C + h * hd;
        float s = 0;
        for (int64_t d = 0; d < hd; d++) s += qi[d] * kj[d];
        s *= scale;
        if (causal && j > i) s = -INFINITY;
        sc[j] = s;
        mx = std::max(mx, s);
      }
      float sum = 0;
      for (int64_t j = 0; j < Lk; j++) {
        sc[j] = std::exp(sc[j] - mx);
        sum += sc[j];
      }
      float* o = op + (int64_t)i * C + h * hd;
      for (int64_t d = 0; d < hd; d++) o[d] = 0;
      for (int64_t j = 0; j < Lk; j++) {
        float wgt = sc[j] / sum;
        const float* vj = vp + j * C + h * hd;
        for (int64_t d = 0; d < hd; d++) o[d] += wgt * vj[d];
      }
    }
  });
}

void SelfAttn::forward(const Tensor& x, const RopeFreqs& fr, Tensor& out) const {
  int64_t L = x.rows(), C = x.cols();
  Tensor qkv;
  linear(x, this->qkv, qkv);  // [L, 3C], each row q|k|v interleaved
  const float* p = qkv.ptr<float>();
  Tensor q({L, C}, DType::F32), k({L, C}, DType::F32), v({L, C}, DType::F32);
  float* qp = q.ptr<float>();
  float* kp = k.ptr<float>();
  float* vp = v.ptr<float>();
  parallel_for(0, L, 1, [&](int64_t bb, int64_t ee) {
    for (int64_t l = bb; l < ee; l++) {
      const float* r = p + l * 3 * C;
      memcpy(qp + l * C, r, C * 4);
      memcpy(kp + l * C, r + C, C * 4);
      memcpy(vp + l * C, r + 2 * C, C * 4);
    }
  });
  rmsnorm(q, nq_w, eps);
  rmsnorm(k, nk_w, eps);
  apply_rope(q, fr);
  apply_rope(k, fr);
  Tensor attn;
  sdpa(q, k, v, Config::num_heads, attn);
  linear(attn, o, out);
}

void CrossAttn::forward(const Tensor& x, const Tensor& ctx, Tensor& out) const {
  Tensor q, k, v;
  linear(x, this->q, q);
  linear(ctx, this->k, k);
  linear(ctx, this->v, v);
  rmsnorm(q, nq_w, eps);
  rmsnorm(k, nk_w, eps);
  Tensor attn;
  sdpa(q, k, v, Config::num_heads, attn);
  linear(attn, o, out);
}

// GELU tanh approximation (nn.GELU(approximate='tanh'), model.py:421).
static void gelu_tanh_inplace(Tensor& x) {
  int64_t n = x.numel();
  float* p = x.ptr<float>();
  parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) {
      float v = p[i];
      float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
      p[i] = 0.5f * v * (1.f + std::tanh(u));
    }
  });
}

void FFN::forward(const Tensor& x, Tensor& y) const {
  Tensor h;
  linear(x, fc1, h);
  gelu_tanh_inplace(h);
  linear(h, fc2, y);
}

// out = a + b
static void add(const Tensor& a, const Tensor& b, Tensor& out) {
  out = Tensor(a.shape, DType::F32);
  int64_t n = a.numel();
  const float* pa = a.ptr<float>();
  const float* pb = b.ptr<float>();
  float* po = out.ptr<float>();
  parallel_for(0, n, 4096, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) po[i] = pa[i] + pb[i];
  });
}

// out = res + y * gate
static void gated_add(const Tensor& res, const Tensor& y, const Tensor& gate,
                      Tensor& out) {
  out = Tensor(res.shape, DType::F32);
  int64_t n = res.numel();
  const float* rp = res.ptr<float>();
  const float* yp = y.ptr<float>();
  const float* gp = gate.ptr<float>();
  float* op = out.ptr<float>();
  parallel_for(0, n, 4096, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) op[i] = rp[i] + yp[i] * gp[i];
  });
}

void mod_shift_scale(const Tensor& ln_x, const Tensor& shift,
                     const Tensor& scale, Tensor& y) {
  // y = ln_x * (1 + scale) + shift  (model.py:456,465)
  y = Tensor(ln_x.shape, DType::F32);
  int64_t n = ln_x.numel();
  const float* xp = ln_x.ptr<float>();
  const float* sp = scale.ptr<float>();
  const float* hp = shift.ptr<float>();
  float* yp = y.ptr<float>();
  parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; i++) yp[i] = xp[i] * (1.f + sp[i]) + hp[i];
  });
}

void TransformerBlock::forward(const Tensor& x, const Tensor& e,
                               const RopeFreqs& fr, const Tensor& ctx,
                               Tensor& out) const {
  int64_t L = x.rows(), C = x.cols();
  // e [L,6,C] + mod_bias [6,C] (ModulationAdd, model.py:368-374,451)
  const float* ep = e.ptr<float>();
  const float* mp = mod_bias.ptr<float>();
  Tensor e0({L, C}, DType::F32), e1({L, C}, DType::F32), e2({L, C}, DType::F32),
      e3({L, C}, DType::F32), e4({L, C}, DType::F32), e5({L, C}, DType::F32);
  Tensor* es[6] = {&e0, &e1, &e2, &e3, &e4, &e5};
  parallel_for(0, L, 1, [&](int64_t bb, int64_t ee) {
    for (int64_t l = bb; l < ee; l++)
      for (int i = 0; i < 6; i++) {
        const float* src = ep + (l * 6 + i) * C;
        const float* bias = mp + (int64_t)i * C;
        float* dst = es[i]->ptr<float>() + l * C;
        for (int64_t c = 0; c < C; c++) dst[c] = src[c] + bias[c];
      }
  });

  // self-attention branch (model.py:454-459)
  Tensor ln1, h, sa, x2;
  layernorm(x, ln1, nullptr, nullptr, eps);
  mod_shift_scale(ln1, e0, e1, h);
  self_attn.forward(h, fr, sa);
  gated_add(x, sa, e2, x2);  // x + sa*e2

  // cross-attention branch, unmodulated (model.py:462-463)
  Tensor ln3, ca, x3;
  layernorm(x2, ln3, &ln3_w, &ln3_b, eps);
  cross_attn.forward(ln3, ctx, ca);
  add(x2, ca, x3);

  // ffn branch (model.py:464-467)
  Tensor ln2, h2, y;
  layernorm(x3, ln2, nullptr, nullptr, eps);
  mod_shift_scale(ln2, e3, e4, h2);
  ffn.forward(h2, y);
  gated_add(x3, y, e5, out);  // x3 + y*e5
}

}  // namespace blk
}  // namespace sd
