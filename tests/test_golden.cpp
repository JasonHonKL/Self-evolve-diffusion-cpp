// Golden parity tests for sd core ops against tests/golden/basic.npz.
// Run from repo root: python3 tools/golden/gen_basic.py && ./build/tests/test_golden
#include "core/tensor.h"
#include "test_support/npz.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace sd;

static Tensor dup(const Tensor& t) {
  Tensor o(t.shape, t.dtype);
  memcpy(o.data, t.data, t.nbytes);
  return o;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) return 1e30f;
  float m = 0;
  const float* x = a.ptr<float>(); const float* y = b.ptr<float>();
  for (int64_t i = 0; i < a.numel(); i++) m = std::max(m, std::fabs(x[i] - y[i]));
  return m;
}

static int fails = 0;

static void check(const char* name, bool ok, float diff, float tol) {
  printf("%-24s %s (max_diff=%g tol=%g)\n", name, ok ? "PASS" : "FAIL", diff, tol);
  if (!ok) fails++;
}

int main() {
  std::unordered_map<std::string, Tensor> g;
  try { g = load_npz("tests/golden/basic.npz"); }
  catch (const std::exception& e) { printf("load_npz: %s\n", e.what()); return 1; }

  {  // rmsnorm in-place
    Tensor x = dup(g.at("rms_x"));
    rmsnorm(x, g.at("rms_w"), 1e-6f);
    float d = max_abs_diff(x, g.at("rms_out"));
    check("rmsnorm", d < 1e-4f, d, 1e-4f);
  }
  {  // silu in-place
    Tensor x = dup(g.at("silu_x"));
    silu_inplace(x);
    float d = max_abs_diff(x, g.at("silu_out"));
    check("silu_inplace", d < 1e-4f, d, 1e-4f);
  }
  {  // matmul
    Tensor C;
    matmul(g.at("mm_a"), g.at("mm_b"), C);
    float d = max_abs_diff(C, g.at("mm_out"));
    check("matmul", d < 1e-4f, d, 1e-4f);
  }
  {  // bf16 roundtrip
    Tensor bf, rt;
    to_bf16(g.at("bf16_x"), bf);
    to_f32(bf, rt);
    float d = max_abs_diff(rt, g.at("bf16_rt"));
    check("bf16_roundtrip", d < 0.008f, d, 0.008f);
  }
  {  // quantize_rowwise: int8 exact, scale tight
    Tensor Wq, sc;
    quantize_rowwise(g.at("q_w"), Wq, sc);
    bool eq = Wq.shape == g.at("q_wq").shape &&
              memcmp(Wq.data, g.at("q_wq").data, Wq.nbytes) == 0;
    float d = max_abs_diff(sc, g.at("q_scale"));
    check("quantize_rowwise.i8", eq, eq ? 0.f : 1e30f, 0);
    check("quantize_rowwise.scale", d < 1e-6f, d, 1e-6f);
  }
  {  // qgemm: cosine vs fp32 reference > 0.999 (dynamic act quant semantics)
    Tensor O;
    qgemm(g.at("qg_a"), g.at("qg_bq"), g.at("qg_bs"), O);
    const float* o = O.ptr<float>(); const float* r = g.at("qg_out").ptr<float>();
    int64_t n = O.numel(); double dot = 0, no = 0, nr = 0;
    for (int64_t i = 0; i < n; i++) { dot += (double)o[i]*r[i]; no += (double)o[i]*o[i]; nr += (double)r[i]*r[i]; }
    float cos = (float)(dot / (std::sqrt(no) * std::sqrt(nr) + 1e-30));
    check("qgemm.cosine", cos > 0.999f, cos, 0.999f);
  }

  if (fails) { printf("%d FAILURES\n", fails); return 1; }
  printf("ALL GOLDEN TESTS PASSED\n");
  return 0;
}
