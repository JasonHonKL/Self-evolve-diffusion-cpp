// UMT5-XXL encoder parity test vs torch golden (issue #9).
// Run from repo root:
//   python3 tools/convert_t5.py && python3 tools/golden/gen_t5_golden.py
//   ./build/tests/test_t5enc
// Pass: per-prompt cosine > 0.98 over the [L,4096] hidden states (bf16 torch
// golden vs bf16-weights/fp32-accum C++ — exact parity impossible).
#include "model/t5enc.h"
#include "test_support/npz.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

using namespace sd;

int main() {
  std::unordered_map<std::string, Tensor> g;
  try { g = load_npz("tools/golden/t5_golden.npz"); }
  catch (const std::exception& e) { printf("load_npz: %s\n", e.what()); return 1; }

  t5::T5Encoder enc("ckpts/t5_enc.safetensors");
  const Tensor& ids = g.at("ids");      // [B, L]
  const Tensor& mask = g.at("mask");    // [B, L]
  const Tensor& hidden = g.at("hidden");  // [B, L, 4096] f32
  const int64_t B = ids.dim(0), L = ids.dim(1), D = t5::T5Encoder::dim;

  int fails = 0;
  for (int64_t b = 0; b < B; b++) {
    Tensor id1({L}, DType::F32), m1({L}, DType::F32);
    memcpy(id1.data, ids.ptr<float>() + b * L, (size_t)L * 4);
    memcpy(m1.data, mask.ptr<float>() + b * L, (size_t)L * 4);

    auto t0 = std::chrono::steady_clock::now();
    Tensor out;
    enc.forward(id1, m1, out);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // cosine over all tokens, plus max abs err relative to the golden's RMS
    // (final hidden magnitudes are ~0.01, so elementwise |r| thresholds are
    // meaningless here)
    const float* o = out.ptr<float>();
    const float* r = hidden.ptr<float>() + b * L * D;
    double dot = 0, no = 0, nr = 0, maxabs = 0;
    for (int64_t i = 0; i < L * D; i++) {
      dot += (double)o[i] * r[i];
      no += (double)o[i] * o[i];
      nr += (double)r[i] * r[i];
      maxabs = std::max(maxabs, (double)std::fabs(o[i] - r[i]));
    }
    double cos = dot / (std::sqrt(no) * std::sqrt(nr) + 1e-30);
    double maxrel = maxabs / (std::sqrt(nr / (L * D)) + 1e-30);
    bool ok = cos > 0.98;
    printf("prompt %lld (L=%lld): cosine=%.6f max_rel=%.4f  %s  [%.0f ms]\n",
           (long long)b, (long long)L, cos, maxrel, ok ? "PASS" : "FAIL", ms);
    if (!ok) fails++;
  }
  if (fails) { printf("T5ENC TEST FAILED\n"); return 1; }
  printf("T5ENC TEST PASSED\n");
  return 0;
}
