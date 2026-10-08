// Vocoder parity test vs torch fp32 golden (issue #13).
// Prereqs (from repo root):
//   python3 tools/convert_vocoder.py
//   python3 tools/golden/gen_voc_golden.py
//   ./build/tests/test_vocoder
// Pass: cosine > 0.98 at each stage (mel / wav / full chain); prints timing
// at Ovi's real latent lengths (T=157 = 5s, T=314 = 10s).
#include "model/vocoder.h"
#include "test_support/npz.h"
#include <chrono>
#include <cmath>
#include <cstdio>

using namespace sd;

static double cosine(const Tensor& a, const Tensor& b) {
  const float* x = a.ptr<float>();
  const float* y = b.ptr<float>();
  double dot = 0, nx = 0, ny = 0;
  for (int64_t i = 0; i < a.numel(); i++) {
    dot += (double)x[i] * y[i];
    nx += (double)x[i] * x[i];
    ny += (double)y[i] * y[i];
  }
  return dot / (std::sqrt(nx) * std::sqrt(ny) + 1e-30);
}

int main() {
  std::unordered_map<std::string, Tensor> g;
  try { g = load_npz("tests/golden/voc_golden.npz"); }
  catch (const std::exception& e) { printf("load_npz: %s\n", e.what()); return 1; }

  voc::Vocoder vocoder("ckpts/mmaudio_vocoder.safetensors");
  int fails = 0;

  // stage parity
  Tensor mel, wav;
  vocoder.decode_latent(g.at("z"), mel);
  double c_mel = cosine(mel, g.at("mel"));
  printf("mel  [80,%lld]: cosine=%.6f  %s\n", (long long)mel.dim(1), c_mel,
         c_mel > 0.98 ? "PASS" : "FAIL");
  if (c_mel <= 0.98) fails++;

  vocoder.vocode(g.at("mel"), wav);
  double c_wav = cosine(wav, g.at("wav"));
  printf("wav  [1,%lld]: cosine=%.6f  %s\n", (long long)wav.numel(), c_wav,
         c_wav > 0.98 ? "PASS" : "FAIL");
  if (c_wav <= 0.98) fails++;

  Tensor full;
  vocoder.latent_to_wav(g.at("z"), full);
  double c_full = cosine(full, g.at("wav"));
  printf("full [1,%lld]: cosine=%.6f  %s\n", (long long)full.numel(), c_full,
         c_full > 0.98 ? "PASS" : "FAIL");
  if (c_full <= 0.98) fails++;

  // timing at Ovi's real latent lengths (5s and 10s clips)
  for (int64_t T : {157, 314}) {
    Tensor z({20, T}, DType::F32);
    randu(z, 7 + T);
    Tensor m, w;
    auto t0 = std::chrono::steady_clock::now();
    vocoder.decode_latent(z, m);
    auto t1 = std::chrono::steady_clock::now();
    vocoder.vocode(m, w);
    auto t2 = std::chrono::steady_clock::now();
    double dms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double vms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("T=%lld latents (%.1fs audio): vae %.0f ms + bigvgan %.0f ms = %.0f ms (%.1fx realtime)\n",
           (long long)T, T / 31.25, dms, vms, dms + vms, (T / 31.25) / ((dms + vms) / 1e3));
  }

  if (fails) { printf("VOCODER TEST FAILED\n"); return 1; }
  printf("VOCODER TEST PASSED\n");
  return 0;
}
