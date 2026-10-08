// Implementation for src/model/ovi_dit.h. Reuses sd::/blk:: primitives
// (linear/layernorm/rmsnorm/apply_rope/sdpa/rope_freqs_*); blk::SelfAttn &
// friends hardcode Config::num_heads so block forwards are re-implemented
// here with a heads parameter.
#include "model/ovi_dit.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sd {
namespace ovi {
namespace {

constexpr float kEps = 1e-6f;

Tensor transp2d(const Tensor& w) {  // [m,n] -> [n,m]
  int64_t m = w.rows(), n = w.cols();
  Tensor t({n, m}, DType::F32);
  const float* s = w.ptr<float>();
  float* d = t.ptr<float>();
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < n; j++) d[j * m + i] = s[i * n + j];
  return t;
}

// Contiguous flat view of t (numel == r*c), reinterpreted [r,c]. Caller keeps
// the backing tensor alive.
Tensor flat2d(const Tensor& t, int64_t r, int64_t c) {
  Tensor v;
  v.shape = {r, c};
  v.dtype = DType::F32;
  v.data = t.data;
  v.nbytes = r * c * 4;
  v.owning = false;
  return v;
}

Tensor row_view(const Tensor& t, int64_t row) {  // row slice view [C]
  Tensor v = flat2d(t, t.rows(), t.cols());
  v.data = (char*)t.ptr<float>() + row * t.cols() * 4;
  v.nbytes = t.cols() * 4;
  v.shape = {t.cols()};
  return v;
}

Tensor dup(const Tensor& x) {
  Tensor y(x.shape, DType::F32);
  memcpy(y.ptr<float>(), x.ptr<float>(), x.nbytes);
  return y;
}

// nn.GELU(approximate='tanh') — same closed form as blocks.cpp (static there).
void gelu_tanh_inplace(Tensor& x) {
  int64_t n = x.numel();
  float* p = x.ptr<float>();
  for (int64_t i = 0; i < n; i++) {
    float v = p[i];
    float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    p[i] = 0.5f * v * (1.f + std::tanh(u));
  }
}

void add_inplace(Tensor& x, const Tensor& y) {
  int64_t n = x.numel();
  float* a = x.ptr<float>();
  const float* b = y.ptr<float>();
  for (int64_t i = 0; i < n; i++) a[i] += b[i];
}

// x += y * gate, gate [C] broadcast per row.
void gated_add_inplace(Tensor& x, const Tensor& y, const Tensor& gate) {
  int64_t L = x.rows(), C = x.cols();
  float* xp = x.ptr<float>();
  const float* yp = y.ptr<float>();
  const float* g = gate.ptr<float>();
  for (int64_t l = 0; l < L; l++)
    for (int64_t c = 0; c < C; c++) xp[l * C + c] += yp[l * C + c] * g[c];
}

// y = x * (1 + scale) + shift, scale/shift [C] broadcast (model.py:456,465).
void mod_rows(const Tensor& x, const Tensor& shift, const Tensor& scale,
              Tensor& y) {
  int64_t L = x.rows(), C = x.cols();
  y = Tensor({L, C}, DType::F32);
  const float* xp = x.ptr<float>();
  const float* sp = shift.ptr<float>();
  const float* hp = scale.ptr<float>();
  float* yp = y.ptr<float>();
  for (int64_t l = 0; l < L; l++)
    for (int64_t c = 0; c < C; c++)
      yp[l * C + c] = xp[l * C + c] * (1.f + hp[c]) + sp[c];
}

// Self-attention (model.py:198-265): separate q/k/v, RMSNorm q/k, rope q/k.
void self_attn_fwd(const SelfAttnW& a, const Tensor& x,
                   const blk::RopeFreqs& fr, int heads, Tensor& out) {
  Tensor q, k, v, att;
  blk::linear(x, a.q, q);
  blk::linear(x, a.k, k);
  blk::linear(x, a.v, v);
  rmsnorm(q, a.nq_w, kEps);
  rmsnorm(k, a.nk_w, kEps);
  blk::apply_rope(q, fr, heads);
  blk::apply_rope(k, fr, heads);
  blk::sdpa(q, k, v, heads, att);
  blk::linear(att, a.o, out);
}

// Plain t2v cross-attention on text context, no rope (model.py:268-294).
void cross_text_fwd(const CrossAttnW& a, const Tensor& x, const Tensor& ctx,
                    int heads, Tensor& out) {
  Tensor q, k, v, att;
  blk::linear(x, a.q, q);
  blk::linear(ctx, a.k, k);
  blk::linear(ctx, a.v, v);
  rmsnorm(q, a.nq_w, kEps);
  rmsnorm(k, a.nk_w, kEps);
  blk::sdpa(q, k, v, heads, att);
  blk::linear(att, a.o, out);
}

// Modulated self-attn branch (model.py:454-459). e [6,dim] with ModulationAdd
// bias already applied. In-place on x.
void block_self_fwd(const BlockW& b, Tensor& x, const blk::RopeFreqs& fr,
                    const Tensor& e, int heads) {
  Tensor ln1, h, sa;
  blk::layernorm(x, ln1, nullptr, nullptr, kEps);
  mod_rows(ln1, row_view(e, 0), row_view(e, 1), h);
  self_attn_fwd(b.sa, h, fr, heads, sa);
  gated_add_inplace(x, sa, row_view(e, 2));
}

void ffn_fwd(const BlockW& b, Tensor& x, const Tensor& e) {
  Tensor ln2, h, y;
  blk::layernorm(x, ln2, nullptr, nullptr, kEps);
  mod_rows(ln2, row_view(e, 3), row_view(e, 4), h);
  b.ffn.forward(h, y);
  gated_add_inplace(x, y, row_view(e, 5));
}

// Fused cross-attn + FFN (fusion.py:72-159): text attention on ctx PLUS
// target attention — q roped with src freqs only for the target part
// (fusion.py:103 unroped vs :118 roped), k_t roped with target freqs.
// Residual around the whole o(text + target) sum (fusion.py:145), then the
// usual modulated FFN.
void fusion_cross_ffn(const BlockW& b, Tensor& x, const Tensor& ctx,
                      const Tensor& e, const Tensor& target,
                      const blk::RopeFreqs& fr_src,
                      const blk::RopeFreqs& fr_tgt, int heads) {
  const CrossAttnW& a = b.ca;
  Tensor q, k, v, xt;
  blk::linear(x, a.q, q);
  blk::linear(ctx, a.k, k);
  blk::linear(ctx, a.v, v);
  rmsnorm(q, a.nq_w, kEps);
  rmsnorm(k, a.nk_w, kEps);
  blk::sdpa(q, k, v, heads, xt);

  Tensor tn, kt, vt, xtg;
  blk::layernorm(target, tn, &a.fusion.pnf_w, &a.fusion.pnf_b, kEps);
  blk::linear(tn, a.fusion.k_fusion, kt);
  blk::linear(tn, a.fusion.v_fusion, vt);
  rmsnorm(kt, a.fusion.norm_k_fusion_w, kEps);
  blk::apply_rope(q, fr_src, heads);
  blk::apply_rope(kt, fr_tgt, heads);
  blk::sdpa(q, kt, vt, heads, xtg);

  add_inplace(xt, xtg);
  Tensor co;
  blk::linear(xt, a.o, co);
  add_inplace(x, co);
  ffn_fwd(b, x, e);
}

// Vanilla WanAttentionBlock forward (model.py:430-471).
void block_vanilla_fwd(const BlockW& b, Tensor& x, const Tensor& ctx,
                       const blk::RopeFreqs& fr, const Tensor& e, int heads) {
  block_self_fwd(b, x, fr, e, heads);
  Tensor ln3, ca;
  blk::layernorm(x, ln3, &b.ln3_w, &b.ln3_b, kEps);
  cross_text_fwd(b.ca, ln3, ctx, heads, ca);
  add_inplace(x, ca);  // unmodulated cross residual (model.py:463)
  ffn_fwd(b, x, e);
}

// Conv1d odd-k pad=k/2, w [out,in,k] torch layout, optional bias
// (ChannelLastConv1d, model.py:110-116,618-622).
void conv1d_pad(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& y) {
  int64_t L = x.rows(), Cin = x.cols(), Cout = w.dim(0), K = w.dim(2);
  int64_t P = K / 2;
  y = Tensor({L, Cout}, DType::F32);
  float* yp = y.ptr<float>();
  memset(yp, 0, y.nbytes);
  if (b) {
    const float* bp = b->ptr<float>();
    for (int64_t l = 0; l < L; l++)
      for (int64_t o = 0; o < Cout; o++) yp[l * Cout + o] = bp[o];
  }
  const float* xp = x.ptr<float>();
  const float* wp = w.ptr<float>();
  for (int64_t j = 0; j < K; j++) {
    int64_t off = j - P;  // y[l] += x[l+off] @ w[:,:,j]^T
    int64_t lo = std::max<int64_t>(0, -off);
    int64_t hi = std::min<int64_t>(L, L - off);
    for (int64_t l = lo; l < hi; l++) {
      const float* xr = xp + (l + off) * Cin;
      float* yr = yp + l * Cout;
      for (int64_t o = 0; o < Cout; o++) {
        const float* wr = wp + o * Cin * K + j;  // (o,c,j) at o*Cin*K + c*K + j
        float s = 0;
        for (int64_t c = 0; c < Cin; c++) s += xr[c] * wr[c * K];
        yr[o] += s;
      }
    }
  }
}

// Head (model.py:474-501): x = head(ln(x)*(1+e1)+e0), shift/scale from
// head_mod [2,dim] + time-embedding row e [dim].
void head_fwd(const BackboneW& bb, const Tensor& x, const Tensor& e_row,
              Tensor& out) {
  Tensor ln, shift, scale, h;
  blk::layernorm(x, ln, nullptr, nullptr, kEps);
  shift = dup(row_view(bb.head_mod, 0));
  scale = dup(row_view(bb.head_mod, 1));
  float* sh = shift.ptr<float>();
  float* sc = scale.ptr<float>();
  const float* er = e_row.ptr<float>();
  int64_t C = e_row.numel();
  for (int64_t c = 0; c < C; c++) {
    sh[c] += er[c];
    sc[c] += er[c];
  }
  mod_rows(ln, shift, scale, h);
  blk::linear(h, bb.head, out);
}

}  // namespace

OviDitCfg OviDitCfg::real() {
  OviDitCfg c;
  // video.json: ti2v, patch (1,2,2), in/out 48; audio.json: t2a, in/out 20.
  c.video = BackboneCfg{3072, 14336, 24, 30, 256, 4096, 512, 48, 48, 1, 2, 2, 1.0f};
  c.audio = BackboneCfg{3072, 14336, 24, 30, 256, 4096, 512, 20, 20, 1, 1, 1, 0.19676f};
  c.fusion_layers.resize(30);
  for (int i = 0; i < 30; i++) c.fusion_layers[i] = i;
  return c;
}

OviDitCfg OviDitCfg::test() {
  OviDitCfg c;
  c.video = BackboneCfg{384, 1792, 3, 2, 32, 512, 64, 48, 48, 1, 2, 2, 1.0f};
  c.audio = BackboneCfg{384, 1792, 3, 2, 32, 512, 64, 20, 20, 1, 1, 1, 0.19676f};
  c.fusion_layers = {1};
  return c;
}

void OviDit::init(const OviDitCfg& c) {
  cfg = c;
  video.blocks.resize(c.video.num_layers);
  audio.blocks.resize(c.audio.num_layers);
}

void OviDit::load_quantized_pack(const std::string&) {
  // TODO(#15): pack int8 weights + scales per Linear, route through sd::qgemm.
  throw std::runtime_error(
      "ovi_dit: quantized pack loading not yet implemented (issue #15)");
}

void OviDit::load(const std::unordered_map<std::string, Tensor>& m) {
  auto get = [&](const std::string& n) -> const Tensor& {
    auto it = m.find(n);
    if (it == m.end() || !it->second.data)
      throw std::runtime_error("ovi_dit: missing tensor " + n);
    return it->second;
  };
  // bf16 -> f32 into pool_ (deque: stable refs across growth).
  auto f32 = [&](const Tensor& t) -> const Tensor& {
    if (t.dtype == DType::F32) return t;
    Tensor o;
    to_f32(t, o);
    pool_.push_back(std::move(o));
    return pool_.back();
  };
  auto chk2 = [&](const Tensor& t, const std::string& n, int64_t d0,
                  int64_t d1) {
    if ((int64_t)t.shape.size() != 2 || (int64_t)t.dim(0) != d0 || (int64_t)t.dim(1) != d1)
      throw std::runtime_error("ovi_dit: bad shape for " + n);
  };
  auto chk1 = [&](const Tensor& t, const std::string& n, int64_t d) {
    if ((int64_t)t.shape.size() != 1 || (int64_t)t.dim(0) != d)
      throw std::runtime_error("ovi_dit: bad shape for " + n);
  };
  // torch Linear [out,in] -> blk::Linear [in,out] (+ bias view).
  auto lin = [&](blk::Linear& l, const std::string& base, int64_t in,
                 int64_t out) {
    const Tensor& w = f32(get(base + ".weight"));
    chk2(w, base + ".weight", out, in);
    l.w = transp2d(w);
    const Tensor& b = f32(get(base + ".bias"));
    chk1(b, base + ".bias", out);
    l.b = flat2d(b, out, 1);
    l.b.shape = {out};
  };
  auto vec = [&](Tensor& dst, const std::string& n, int64_t d) {
    const Tensor& t = f32(get(n));
    chk1(t, n, d);
    dst = flat2d(t, d, 1);
    dst.shape = {d};
  };
  auto conv1d = [&](Tensor& dst, const std::string& n, int64_t o, int64_t i) {
    const Tensor& t = f32(get(n));
    if ((int64_t)t.shape.size() != 3 || (int64_t)t.dim(2) != 7 ||
        (o >= 0 && (int64_t)t.dim(0) != o) || (i >= 0 && (int64_t)t.dim(1) != i))
      throw std::runtime_error("ovi_dit: bad shape for " + n);
    dst = flat2d(t, (int64_t)t.dim(0), (int64_t)t.dim(1) * 7);
    dst.shape = {(int64_t)t.dim(0), (int64_t)t.dim(1), (int64_t)7};
  };

  auto load_backbone = [&](BackboneW& bb, const BackboneCfg& bc,
                           const char* pfx) {
    const bool aud = pfx[0] == 'a';
    std::string p(pfx);
    if (aud) {
      conv1d(bb.pe0, p + "patch_embedding.0.weight", bc.dim, bc.in_dim);
      vec(bb.pe_b, p + "patch_embedding.0.bias", bc.dim);
      // ConvMLP hidden = round256(int(2*4*dim/3)) (model.py:144-145)
      conv1d(bb.ac_w1, p + "patch_embedding.2.w1.weight", -1, bc.dim);
      conv1d(bb.ac_w2, p + "patch_embedding.2.w2.weight", bc.dim, -1);
      conv1d(bb.ac_w3, p + "patch_embedding.2.w3.weight", -1, bc.dim);
    } else {
      const Tensor& w = f32(get(p + "patch_embedding.weight"));
      int64_t pp = bc.pt * bc.ph * bc.pw;
      if ((int64_t)w.shape.size() != 5 || (int64_t)w.dim(0) != bc.dim ||
          (int64_t)w.dim(1) != bc.in_dim || (int64_t)w.dim(2) != bc.pt ||
          (int64_t)w.dim(3) != bc.ph || (int64_t)w.dim(4) != bc.pw)
        throw std::runtime_error("ovi_dit: bad video patch_embedding");
      bb.pe_mat = transp2d(flat2d(w, bc.dim, bc.in_dim * pp));
      vec(bb.pe_b, p + "patch_embedding.bias", bc.dim);
    }
    lin(bb.te0, p + "text_embedding.0", bc.text_dim, bc.dim);
    lin(bb.te2, p + "text_embedding.2", bc.dim, bc.dim);
    lin(bb.tm0, p + "time_embedding.0", bc.freq_dim, bc.dim);
    lin(bb.tm2, p + "time_embedding.2", bc.dim, bc.dim);
    lin(bb.tp1, p + "time_projection.1", bc.dim, 6 * bc.dim);
    int64_t pout = (aud ? 1 : bc.pt * bc.ph * bc.pw) * bc.out_dim;
    lin(bb.head, p + "head.head", bc.dim, pout);
    const Tensor& hm = f32(get(p + "head.modulation"));
    if ((int64_t)hm.shape.size() != 3 || (int64_t)hm.dim(0) != 1 ||
        (int64_t)hm.dim(1) != 2 || (int64_t)hm.dim(2) != bc.dim)
      throw std::runtime_error("ovi_dit: bad head.modulation");
    bb.head_mod = flat2d(hm, 2, bc.dim);
    for (int i = 0; i < bc.num_layers; i++) {
      BlockW& b = bb.blocks[i];
      std::string B = p + "blocks." + std::to_string(i) + ".";
      lin(b.sa.q, B + "self_attn.q", bc.dim, bc.dim);
      lin(b.sa.k, B + "self_attn.k", bc.dim, bc.dim);
      lin(b.sa.v, B + "self_attn.v", bc.dim, bc.dim);
      lin(b.sa.o, B + "self_attn.o", bc.dim, bc.dim);
      vec(b.sa.nq_w, B + "self_attn.norm_q.weight", bc.dim);
      vec(b.sa.nk_w, B + "self_attn.norm_k.weight", bc.dim);
      lin(b.ca.q, B + "cross_attn.q", bc.dim, bc.dim);
      lin(b.ca.k, B + "cross_attn.k", bc.dim, bc.dim);
      lin(b.ca.v, B + "cross_attn.v", bc.dim, bc.dim);
      lin(b.ca.o, B + "cross_attn.o", bc.dim, bc.dim);
      vec(b.ca.nq_w, B + "cross_attn.norm_q.weight", bc.dim);
      vec(b.ca.nk_w, B + "cross_attn.norm_k.weight", bc.dim);
      vec(b.ln3_w, B + "norm3.weight", bc.dim);
      vec(b.ln3_b, B + "norm3.bias", bc.dim);
      lin(b.ffn.fc1, B + "ffn.0", bc.dim, bc.ffn_dim);
      lin(b.ffn.fc2, B + "ffn.2", bc.ffn_dim, bc.dim);
      const Tensor& mo = f32(get(B + "modulation.modulation"));
      if ((int64_t)mo.shape.size() != 3 || (int64_t)mo.dim(0) != 1 ||
          (int64_t)mo.dim(1) != 6 || (int64_t)mo.dim(2) != bc.dim)
        throw std::runtime_error("ovi_dit: bad modulation " + B);
      b.mod_bias = flat2d(mo, 6, bc.dim);
      bool isf = std::find(cfg.fusion_layers.begin(), cfg.fusion_layers.end(),
                           i) != cfg.fusion_layers.end();
      if (isf) {
        lin(b.ca.fusion.k_fusion, B + "cross_attn.k_fusion", bc.dim, bc.dim);
        lin(b.ca.fusion.v_fusion, B + "cross_attn.v_fusion", bc.dim, bc.dim);
        vec(b.ca.fusion.norm_k_fusion_w,
            B + "cross_attn.norm_k_fusion.weight", bc.dim);
        vec(b.ca.fusion.pnf_w,
            B + "cross_attn.pre_attn_norm_fusion.weight", bc.dim);
        vec(b.ca.fusion.pnf_b,
            B + "cross_attn.pre_attn_norm_fusion.bias", bc.dim);
      }
    }
  };
  load_backbone(video, cfg.video, "video_model.");
  load_backbone(audio, cfg.audio, "audio_model.");
}

std::vector<std::string> OviDit::expected_keys() const {
  std::vector<std::string> ks;
  auto backbone = [&](const BackboneCfg& bc, const char* pfx) {
    const bool aud = pfx[0] == 'a';
    std::string p(pfx);
    if (aud) {
      for (const char* s : {"patch_embedding.0.weight", "patch_embedding.0.bias",
                            "patch_embedding.2.w1.weight",
                            "patch_embedding.2.w2.weight",
                            "patch_embedding.2.w3.weight"})
        ks.push_back(p + s);
    } else {
      ks.push_back(p + "patch_embedding.weight");
      ks.push_back(p + "patch_embedding.bias");
    }
    for (const char* s : {"text_embedding.0.weight", "text_embedding.0.bias",
                          "text_embedding.2.weight", "text_embedding.2.bias",
                          "time_embedding.0.weight", "time_embedding.0.bias",
                          "time_embedding.2.weight", "time_embedding.2.bias",
                          "time_projection.1.weight", "time_projection.1.bias",
                          "head.head.weight", "head.head.bias",
                          "head.modulation"})
      ks.push_back(p + s);
    for (int i = 0; i < bc.num_layers; i++) {
      std::string B = p + "blocks." + std::to_string(i) + ".";
      for (const char* w : {"q", "k", "v", "o"}) {
        ks.push_back(B + "self_attn." + w + ".weight");
        ks.push_back(B + "self_attn." + w + ".bias");
        ks.push_back(B + "cross_attn." + w + ".weight");
        ks.push_back(B + "cross_attn." + w + ".bias");
      }
      for (const char* s : {"self_attn.norm_q.weight", "self_attn.norm_k.weight",
                            "cross_attn.norm_q.weight",
                            "cross_attn.norm_k.weight", "norm3.weight",
                            "norm3.bias", "ffn.0.weight", "ffn.0.bias",
                            "ffn.2.weight", "ffn.2.bias",
                            "modulation.modulation"})
        ks.push_back(B + s);
      bool isf = std::find(cfg.fusion_layers.begin(), cfg.fusion_layers.end(),
                           i) != cfg.fusion_layers.end();
      if (isf)
        for (const char* s : {"cross_attn.k_fusion.weight",
                              "cross_attn.k_fusion.bias",
                              "cross_attn.v_fusion.weight",
                              "cross_attn.v_fusion.bias",
                              "cross_attn.pre_attn_norm_fusion.weight",
                              "cross_attn.pre_attn_norm_fusion.bias",
                              "cross_attn.norm_k_fusion.weight"})
          ks.push_back(B + s);
    }
  };
  backbone(cfg.video, "video_model.");
  backbone(cfg.audio, "audio_model.");
  return ks;
}

void OviDit::forward(const Tensor& nv, const Tensor& na, float t,
                     const Tensor& vid_ctx, const Tensor& aud_ctx,
                     Tensor& out_video, Tensor& out_audio) const {
  const BackboneCfg& vc = cfg.video;
  const BackboneCfg& ac = cfg.audio;
  const int vh = vc.num_heads, ah = ac.num_heads;
  const int vhd = vc.dim / vh, ahd = ac.dim / ah;

  // ---- video patch embed: im2col (feature order c,pt,ph,pw) @ pe_mat
  const int64_t C = nv.dim(0), F = nv.dim(1), H = nv.dim(2), W = nv.dim(3);
  const int64_t Hp = H / vc.ph, Wp = W / vc.pw, Lv = F * Hp * Wp;
  const int64_t pp = vc.pt * vc.ph * vc.pw;
  Tensor cols({Lv, C * pp}, DType::F32);
  {
    const float* src = nv.ptr<float>();
    float* cp = cols.ptr<float>();
    for (int64_t f = 0; f < F; f++)
      for (int64_t h = 0; h < Hp; h++)
        for (int64_t w = 0; w < Wp; w++)
          for (int64_t c = 0; c < C; c++)
            for (int64_t ip = 0; ip < vc.pt; ip++)
              for (int64_t iq = 0; iq < vc.ph; iq++)
                for (int64_t ir = 0; ir < vc.pw; ir++) {
                  int64_t tok = (f * Hp + h) * Wp + w;
                  int64_t feat = ((c * vc.pt + ip) * vc.ph + iq) * vc.pw + ir;
                  cp[tok * (C * pp) + feat] =
                      src[((c * F + (f * vc.pt + ip)) * H + (h * vc.ph + iq)) *
                              W +
                          (w * vc.pw + ir)];
                }
  }
  Tensor xv;  // [Lv, dim]
  matmul(cols, video.pe_mat, xv, video.pe_b.ptr<float>());

  // ---- audio patch embed: Conv1d k7 p3 -> SiLU -> ConvMLP (model.py:616-622)
  const int64_t La = na.rows();
  Tensor h1, xa;
  conv1d_pad(na, audio.pe0, &audio.pe_b, h1);
  silu_inplace(h1);
  {
    Tensor a1, a3;
    conv1d_pad(h1, audio.ac_w1, nullptr, a1);
    silu_inplace(a1);
    conv1d_pad(h1, audio.ac_w3, nullptr, a3);
    int64_t n = a1.numel();
    for (int64_t i = 0; i < n; i++) a1.ptr<float>()[i] *= a3.ptr<float>()[i];
    conv1d_pad(a1, audio.ac_w2, nullptr, xa);
  }

  // ---- rope freqs (video 3D over F,Hp,Wp; audio 1D scaled)
  blk::RopeFreqs vfr = blk::rope_freqs_3d((int)F, (int)Hp, (int)Wp, vhd);
  blk::RopeFreqs afr =
      blk::rope_freqs_1d(La, ahd, blk::Config::rope_theta, ac.rope_scaling);

  // ---- text context: embed + zero-pad to text_len. Padded rows ARE
  //      attended (model.py:784-789 pads with zeros, no context mask).
  auto embed_ctx = [&](const BackboneW& bb, const Tensor& ctx,
                       int64_t text_len) {
    Tensor e0, e;
    blk::linear(ctx, bb.te0, e0);
    gelu_tanh_inplace(e0);
    blk::linear(e0, bb.te2, e);
    Tensor full({text_len, e.cols()}, DType::F32);
    memset(full.ptr<float>(), 0, full.nbytes);
    int64_t rows = std::min<int64_t>(text_len, e.rows());
    memcpy(full.ptr<float>(), e.ptr<float>(), rows * e.cols() * 4);
    return full;
  };

  // ---- timestep: sinusoidal -> MLP -> e [1,dim]; time_projection -> [6,dim].
  //      Scalar t replicates over all tokens (model.py:734-749); per-token t
  //      (first_frame_is_clean) is a sampler concern, not needed here.
  auto time_rows = [&](const BackboneW& bb, const BackboneCfg& bc, Tensor& e,
                       Tensor& e6) {
    int64_t half = bc.freq_dim / 2;
    Tensor pos({1, bc.freq_dim}, DType::F32);
    float* pf = pos.ptr<float>();
    for (int64_t j = 0; j < half; j++) {
      double f = std::pow(10000.0, -(double)j / (double)half);
      pf[j] = (float)std::cos(t * f);  // cat([cos, sin]) — model.py:24-34
      pf[half + j] = (float)std::sin(t * f);
    }
    blk::linear(pos, bb.tm0, e);
    silu_inplace(e);
    Tensor e2;
    blk::linear(e, bb.tm2, e2);
    e = std::move(e2);
    // time_projection = SiLU + Linear (model.py:633) — head keeps RAW e
    Tensor es = dup(e);
    silu_inplace(es);
    Tensor six;
    blk::linear(es, bb.tp1, six);  // [1, 6*dim]
    e6 = Tensor({6, (int64_t)bc.dim}, DType::F32);
    memcpy(e6.ptr<float>(), six.ptr<float>(), e6.nbytes);
  };

  // ---- layer stack (fusion.py:286-302)
  std::vector<char> is_fusion(cfg.video.num_layers, 0);
  for (int i : cfg.fusion_layers)
    if (i < cfg.video.num_layers) is_fusion[i] = 1;
  Tensor ve, ve6, ae, ae6;
  {
    Tensor vfull = embed_ctx(video, vid_ctx, vc.text_len),
           afull = embed_ctx(audio, aud_ctx, ac.text_len);
    time_rows(video, vc, ve, ve6);
    time_rows(audio, ac, ae, ae6);
    for (int i = 0; i < cfg.video.num_layers; i++) {
      if (cfg.slg_layer > 0 && i == cfg.slg_layer) continue;  // fusion.py:290
      const BlockW& vb = video.blocks[i];
      const BlockW& ab = audio.blocks[i];
      Tensor e_v = dup(ve6), e_a = dup(ae6);  // + per-block ModulationAdd
      {
        float* pv = e_v.ptr<float>();
        const float* bv = vb.mod_bias.ptr<float>();
        float* pa = e_a.ptr<float>();
        const float* ba = ab.mod_bias.ptr<float>();
        int64_t n = (int64_t)6 * vc.dim;
        for (int64_t k = 0; k < n; k++) pv[k] += bv[k];
        for (int64_t k = 0; k < n; k++) pa[k] += ba[k];
      }
      if (is_fusion[i]) {
        // single_fusion_block_forward, fusion.py:161-240: audio self-attn,
        // video self-attn, audio cross+ffn (target=video), video cross+ffn
        // (target=og_audio = audio state PRE cross).
        block_self_fwd(ab, xa, afr, e_a, ah);
        block_self_fwd(vb, xv, vfr, e_v, vh);
        Tensor og_audio = dup(xa);
        fusion_cross_ffn(ab, xa, afull, e_a, xv, afr, vfr, ah);
        fusion_cross_ffn(vb, xv, vfull, e_v, og_audio, vfr, afr, vh);
      } else {
        block_vanilla_fwd(ab, xa, afull, afr, e_a, ah);
        block_vanilla_fwd(vb, xv, vfull, vfr, e_v, vh);
      }
    }
  }

  // ---- heads + unpatchify (model.py:474-501, 806-822, 880-905)
  Tensor hv, ha;
  head_fwd(video, xv, ve, hv);  // [Lv, pp*out_dim], feature order pt,ph,pw,C
  head_fwd(audio, xa, ae, ha);  // [La, out_dim]
  out_video = Tensor({vc.out_dim, F, H, W}, DType::F32);
  {
    const float* s = hv.ptr<float>();
    float* d = out_video.ptr<float>();
    for (int64_t f = 0; f < F; f++)
      for (int64_t h = 0; h < Hp; h++)
        for (int64_t w = 0; w < Wp; w++)
          for (int64_t ip = 0; ip < vc.pt; ip++)
            for (int64_t iq = 0; iq < vc.ph; iq++)
              for (int64_t ir = 0; ir < vc.pw; ir++)
                for (int64_t c = 0; c < vc.out_dim; c++) {
                  int64_t tok = (f * Hp + h) * Wp + w;
                  int64_t feat =
                      ((ip * vc.ph + iq) * vc.pw + ir) * vc.out_dim + c;
                  int64_t dst = ((c * F + (f * vc.pt + ip)) * H +
                                 (h * vc.ph + iq)) * W +
                                (w * vc.pw + ir);
                  d[dst] = s[tok * hv.cols() + feat];
                }
  }
  out_audio = Tensor({La, ac.out_dim}, DType::F32);
  memcpy(out_audio.ptr<float>(), ha.ptr<float>(), ha.nbytes);
}

}  // namespace ovi
}  // namespace sd
