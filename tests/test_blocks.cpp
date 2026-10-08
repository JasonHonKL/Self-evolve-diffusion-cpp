// Golden parity test for src/model/blocks.{h,cpp} against tools/golden/blocks.npz.
// Run `python3 tools/golden/gen_blocks.py` first. fp32 rel-err < 1e-3 per component.
#include "model/blocks.h"
#include "test_support/npz.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace sd;
using blk::Config;

static int g_fails = 0;

static double relerr(const Tensor& a, const Tensor& b) {
  int64_t n = a.numel();
  const float* pa = a.ptr<float>();
  const float* pb = b.ptr<float>();
  double ds = 0, bs = 0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double)pa[i] - pb[i];
    ds += d * d;
    bs += (double)pb[i] * pb[i];
  }
  return std::sqrt(ds) / (std::sqrt(bs) + 1e-12);
}

static void check(const char* name, const Tensor& got, const Tensor& want) {
  double e = relerr(got, want);
  bool ok = e < 1e-3;
  if (!ok) g_fails++;
  std::printf("%-28s rel-err %.3e  %s\n", name, e, ok ? "OK" : "FAIL");
}

static Tensor dup(const Tensor& x) {
  Tensor y(x.shape, DType::F32);
  memcpy(y.ptr<float>(), x.ptr<float>(), x.nbytes);
  return y;
}

int main() {
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/blocks.npz");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "load_npz: %s (run python3 tools/golden/gen_blocks.py)\n", ex.what());
    return 1;
  }
  auto take = [&](const char* n) -> Tensor {
    auto it = g.find(n);
    if (it == g.end() || !it->second.data) {
      std::fprintf(stderr, "missing/already-taken array %s\n", n);
      exit(1);
    }
    return std::move(it->second);
  };

  const int64_t L = 64, DIM = Config::dim;

  // ---- rope: freq computation + apply, video 3D and audio 1D ----
  auto vfr = blk::rope_freqs_3d(4, 4, 4);
  check("rope_freqs_3d", vfr.ang, take("rope_v_ang"));
  auto afr = blk::rope_freqs_1d(L, Config::head_dim, Config::rope_theta,
                                Config::audio_rope_scaling);
  check("rope_freqs_1d", afr.ang, take("rope_a_ang"));
  Tensor rx = take("rope_x");
  Tensor rv = dup(rx);
  blk::apply_rope(rv, vfr);
  check("apply_rope_3d", rv, take("rope_v_out"));
  blk::apply_rope(rx, afr);
  check("apply_rope_1d", rx, take("rope_a_out"));

  // ---- self_attn ----
  blk::SelfAttn sa;
  sa.init(DIM);
  sa.qkv.w = take("sa_wqkv"); sa.qkv.b = take("sa_bqkv");
  sa.nq_w = take("sa_nq");    sa.nk_w = take("sa_nk");
  sa.o.w = take("sa_wo");     sa.o.b = take("sa_bo");
  Tensor sa_out;
  sa.forward(take("sa_x"), vfr, sa_out);
  check("self_attn", sa_out, take("sa_out"));

  // ---- cross_attn ----
  blk::CrossAttn ca;
  ca.init(DIM);
  ca.q.w = take("ca_wq"); ca.q.b = take("ca_bq");
  ca.k.w = take("ca_wk"); ca.k.b = take("ca_bk");
  ca.v.w = take("ca_wv"); ca.v.b = take("ca_bv");
  ca.o.w = take("ca_wo"); ca.o.b = take("ca_bo");
  ca.nq_w = take("ca_nq"); ca.nk_w = take("ca_nk");
  Tensor ctx = take("ca_ctx");
  Tensor ca_out;
  ca.forward(take("ca_x"), ctx, ca_out);
  check("cross_attn", ca_out, take("ca_out"));

  // ---- ffn ----
  blk::FFN ffn;
  ffn.init(DIM, Config::ffn_dim);
  ffn.fc1.w = take("ffn_w1"); ffn.fc1.b = take("ffn_b1");
  ffn.fc2.w = take("ffn_w2"); ffn.fc2.b = take("ffn_b2");
  Tensor ffn_out;
  ffn.forward(take("ffn_x"), ffn_out);
  check("ffn", ffn_out, take("ffn_out"));

  // ---- modulation apply: x + (ln(x)*(1+e4)+e3)*e5 ----
  Tensor mod_x = take("mod_x"), mod_e = take("mod_e");
  Tensor e3({L, DIM}, DType::F32), e4({L, DIM}, DType::F32), e5({L, DIM}, DType::F32);
  for (int64_t l = 0; l < L; l++) {
    memcpy(e3.ptr<float>() + l * DIM, mod_e.ptr<float>() + (l * 6 + 3) * DIM, DIM * 4);
    memcpy(e4.ptr<float>() + l * DIM, mod_e.ptr<float>() + (l * 6 + 4) * DIM, DIM * 4);
    memcpy(e5.ptr<float>() + l * DIM, mod_e.ptr<float>() + (l * 6 + 5) * DIM, DIM * 4);
  }
  Tensor ln, mh, mod_out;
  blk::layernorm(mod_x, ln, nullptr, nullptr, Config::eps);
  blk::mod_shift_scale(ln, e3, e4, mh);
  Tensor mod_y(mod_x.shape, DType::F32);
  {
    const float* xp = mod_x.ptr<float>(); const float* hp = mh.ptr<float>();
    const float* gp = e5.ptr<float>(); float* op = mod_y.ptr<float>();
    int64_t n = mod_x.numel();
    parallel_for(0, n, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; i++) op[i] = xp[i] + hp[i] * gp[i];
    });
  }
  check("modulation_apply", mod_y, take("mod_out"));

  // ---- composite transformer block ----
  blk::TransformerBlock blk;
  blk.init(DIM, Config::ffn_dim);
  blk.ln3_w = take("ln3_w"); blk.ln3_b = take("ln3_b");
  blk.mod_bias = take("mod_bias");
  blk.self_attn = std::move(sa);
  blk.cross_attn = std::move(ca);
  blk.ffn = std::move(ffn);
  Tensor blk_out;
  blk.forward(take("blk_x"), take("blk_e"), vfr, ctx, blk_out);
  check("transformer_block", blk_out, take("blk_out"));

  if (g_fails) { std::printf("%d FAILURES\n", g_fails); return 1; }
  std::printf("test_blocks OK\n");
  return 0;
}
