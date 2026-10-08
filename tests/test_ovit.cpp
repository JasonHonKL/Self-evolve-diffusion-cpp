// Golden parity test for the Ovi dual-backbone fusion DiT
// (src/model/ovi_dit.{h,cpp}) against tools/golden/ovit_golden.npz, plus —
// when the 23GB checkpoint is fully downloaded — a header-only real-config
// derivation and name-mapping coverage check.
// Run `python3 tools/golden/gen_ovit.py` first.
#include "model/ovi_dit.h"
#include "serde/safetensors.h"
#include "test_support/npz.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sd;

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
  std::printf("%-16s rel-err %.3e  %s\n", name, e, ok ? "OK" : "FAIL");
}

static void check_i(const char* name, int64_t got, int64_t want) {
  bool ok = got == want;
  if (!ok) g_fails++;
  std::printf("%-28s got %-8lld want %-8lld %s\n", name, (long long)got,
              (long long)want, ok ? "OK" : "FAIL");
}

int main() {
  // ---- 1) scaled-down golden parity ----
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/ovit_golden.npz");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "load_npz: %s (run python3 tools/golden/gen_ovit.py)\n",
                 ex.what());
    return 1;
  }
  ovi::OviDit model;
  model.init(ovi::OviDitCfg::test());
  try {
    model.load(g);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "ovi load: %s\n", ex.what());
    return 1;
  }
  float t = g["t"].ptr<float>()[0];
  Tensor ov, oa;
  model.forward(g["in_video"], g["in_audio"], t, g["in_vctx"], g["in_actx"],
                ov, oa);
  check("out_video", ov, g["out_video"]);
  check("out_audio", oa, g["out_audio"]);

  // quantized pack stub must throw (issue #15 wiring)
  bool threw = false;
  try {
    model.load_quantized_pack("noop.sdcpp");
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) {
    g_fails++;
    std::printf("quantized stub did not throw FAIL\n");
  }

  // ---- 2) real checkpoint: header-only config derivation + coverage ----
  const char* ck = "ckpts/Ovi/model.safetensors";
  struct stat st;
  if (stat(ck, &st) != 0 || st.st_size < 23320000000LL) {
    std::printf(
        "checkpoint absent/incomplete (%lld bytes, want ~23320000000) — "
        "header check skipped\n",
        (long long)(stat(ck, &st) == 0 ? st.st_size : -1));
    return g_fails ? 1 : 0;
  }
  // mmap is lazy: building the view map reads ONLY the header JSON, the data
  // region is never paged in below.
  SafetensorsFile f = load_safetensors(ck);
  std::printf("\nreal checkpoint: %s — %zu tensors, %lld bytes\n", ck,
              f.tensors.size(), (long long)st.st_size);

  int max_blk = -1;
  std::vector<int> fusion_pos;
  for (auto& [k, ten] : f.tensors) {
    int bi = -1;
    if (sscanf(k.c_str(), "video_model.blocks.%d.", &bi) == 1 && bi > max_blk)
      max_blk = bi;
    if (k.find(".cross_attn.k_fusion.weight") != std::string::npos) {
      if (sscanf(k.c_str(), "video_model.blocks.%d.", &bi) == 1)
        fusion_pos.push_back(bi);
    }
  }
  std::sort(fusion_pos.begin(), fusion_pos.end());

  ovi::OviDitCfg real = ovi::OviDitCfg::real();
  auto shape2 = [&](const char* n) -> std::vector<int64_t> {
    auto it = f.tensors.find(n);
    if (it == f.tensors.end()) {
      g_fails++;
      std::printf("MISSING key %s\n", n);
      return {};
    }
    return it->second.shape;
  };
  auto s2 = [&](const char* n) { return shape2(n); };

  std::vector<int64_t> q = s2("video_model.blocks.0.self_attn.q.weight");
  std::vector<int64_t> ff = s2("video_model.blocks.0.ffn.0.weight");
  std::vector<int64_t> te = s2("video_model.text_embedding.0.weight");
  std::vector<int64_t> tm = s2("video_model.time_embedding.0.weight");
  std::vector<int64_t> pe = s2("video_model.patch_embedding.weight");
  std::vector<int64_t> vh = s2("video_model.head.head.weight");
  std::vector<int64_t> ape = s2("audio_model.patch_embedding.0.weight");
  std::vector<int64_t> ah = s2("audio_model.head.head.weight");
  std::printf(
      "derived real config: layers=%d dim=%lld ffn=%lld freq_dim=%lld "
      "text_dim=%lld | video: patch=(%lld,%lld,%lld) in=%lld out=%lld | "
      "audio: in=%lld out=%lld | fusion layers=%zu (all: %s) | heads=24 "
      "(from video.json; not in state_dict)\n",
      max_blk + 1, (long long)q[1], (long long)ff[0], (long long)tm[1],
      (long long)te[1], (long long)pe[2], (long long)pe[3], (long long)pe[4],
      (long long)pe[1], (long long)(vh[0] / (pe[2] * pe[3] * pe[4])),
      (long long)ape[1], (long long)ah[0], fusion_pos.size(),
      fusion_pos.size() == (size_t)(max_blk + 1) ? "yes" : "NO");

  check_i("num_layers", max_blk + 1, real.video.num_layers);
  check_i("dim", q[1], real.video.dim);
  check_i("ffn_dim", ff[0], real.video.ffn_dim);
  check_i("freq_dim", tm[1], real.video.freq_dim);
  check_i("text_dim", te[1], real.video.text_dim);
  check_i("video in_dim", pe[1], real.video.in_dim);
  check_i("video out_dim", vh[0] / (pe[2] * pe[3] * pe[4]), real.video.out_dim);
  check_i("audio in_dim", ape[1], real.audio.in_dim);
  check_i("audio out_dim", ah[0], real.audio.out_dim);
  check_i("fusion layer count", (int64_t)fusion_pos.size(),
          (int64_t)real.fusion_layers.size());

  // name-mapping coverage, both directions
  ovi::OviDit realmodel;
  realmodel.init(real);
  std::vector<std::string> exp_keys = realmodel.expected_keys();
  std::unordered_set<std::string> exp(exp_keys.begin(), exp_keys.end());
  std::vector<std::string> uncovered, extra;
  for (auto& [k, ten] : f.tensors)
    if (!exp.count(k)) uncovered.push_back(k);
  for (auto& k : exp)
    if (!f.tensors.count(k)) extra.push_back(k);
  std::sort(uncovered.begin(), uncovered.end());
  std::sort(extra.begin(), extra.end());
  std::printf("coverage: %zu/%zu checkpoint keys mapped; %zu expected keys "
              "absent from checkpoint\n",
              f.tensors.size() - uncovered.size(), f.tensors.size(),
              extra.size());
  for (auto& k : uncovered) std::printf("  UNCOVERED: %s\n", k.c_str());
  for (auto& k : extra) std::printf("  ABSENT:    %s\n", k.c_str());
  if (!uncovered.empty() || !extra.empty()) g_fails++;

  if (g_fails) {
    std::printf("test_ovit: %d FAILURES\n", g_fails);
    return 1;
  }
  std::printf("test_ovit: all passed\n");
  return 0;
}
