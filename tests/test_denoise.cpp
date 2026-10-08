// Denoise-loop test (src/pipeline/denoise.{h,cpp}) on the scaled-down golden
// OviDit (random weights from tools/golden/ovit_golden.npz): 8-step run with
// CFG + SLG(fusion layer 1), finite/shape/determinism asserts, then the numpy
// CFG-step cross-check via tools/golden/check_cfg_step.py on dumped step 0.
// Run `python3 tools/golden/gen_ovit.py` first.
#include "model/ovi_dit.h"
#include "pipeline/denoise.h"
#include "test_support/npz.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sd;

static int g_fails = 0;

static void ck(bool ok, const char* what) {
  if (!ok) g_fails++;
  std::printf("%-42s %s\n", what, ok ? "OK" : "FAIL");
}

static bool all_finite(const Tensor& x) {
  const float* p = x.ptr<float>();
  for (int64_t i = 0; i < x.numel(); i++)
    if (!std::isfinite(p[i])) return false;
  return true;
}

static bool same_bytes(const Tensor& a, const Tensor& b) {
  return a.numel() == b.numel() &&
         std::memcmp(a.ptr<float>(), b.ptr<float>(), a.nbytes) == 0;
}

int main() {
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/ovit_golden.npz");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "load_npz: %s (run python3 tools/golden/gen_ovit.py)\n",
                 ex.what());
    return 1;
  }
  ovi::OviDit dit;
  dit.init(ovi::OviDitCfg::test());
  dit.load(g);  // random weights from the golden gen (rng seed 7)

  // ctx packing [2, Lc, text_dim]: pos rows (in_vctx, in_actx); neg = swap,
  // so the two CFG branches see distinct per-branch contexts.
  const int64_t Lc = g["in_vctx"].dim(0), D = g["in_vctx"].dim(1);
  auto stack2 = [&](const Tensor& a, const Tensor& b) {
    Tensor t({2, Lc, D}, DType::F32);
    std::memcpy(t.ptr<float>(), a.ptr<float>(), a.nbytes);
    std::memcpy((char*)t.ptr<float>() + a.nbytes, b.ptr<float>(), b.nbytes);
    return t;
  };
  Tensor ctx_pos = stack2(g["in_vctx"], g["in_actx"]);
  Tensor ctx_neg = stack2(g["in_actx"], g["in_vctx"]);

  // initial noise: deterministic randu ([-1,1) uniform — the loop math is
  // what's under test, not the noise distribution; engine uses randn)
  auto make_noise = [&](Tensor& xv, Tensor& xa, uint64_t seed) {
    xv = Tensor({48, 4, 8, 8}, DType::F32);
    xa = Tensor({48, 20}, DType::F32);
    randu(xv, seed);
    randu(xa, seed);  // engine: fresh generator per branch, same seed
  };

  DenoiseCfg cfg;
  cfg.steps = 8;
  cfg.video_cfg = 4.0;
  cfg.audio_cfg = 3.0;
  cfg.slg_layer = 1;  // test cfg: layer 1 is the fusion layer — SLG skips it

  // run A: plain (no dump env)
  Tensor xv1, xa1, xv2, xa2;
  make_noise(xv1, xa1, 42);
  int calls = 0;
  ck(denoise(dit, cfg, xv1, xa1, ctx_pos, ctx_neg, 42,
             [&](int i, int n) { calls++; (void)i; (void)n; }),
     "denoise run A returns true");
  ck(calls == cfg.steps, "progress callback fired per step");

  // run B: same seed, dump enabled — must be byte-identical to A
  make_noise(xv2, xa2, 42);
  std::string dump = "/tmp/ovi_cfg_step";
  std::string rm = "rm -f " + dump + "_s*.npy";
  if (std::system(rm.c_str()) != 0) {}
  setenv("OVI_DENOISE_DUMP", dump.c_str(), 1);
  ck(denoise(dit, cfg, xv2, xa2, ctx_pos, ctx_neg, 42, nullptr),
     "denoise run B returns true");
  unsetenv("OVI_DENOISE_DUMP");

  ck(all_finite(xv1) && all_finite(xa1), "outputs finite");
  ck(xv1.shape == std::vector<int64_t>({48, 4, 8, 8}) &&
     xa1.shape == std::vector<int64_t>({48, 20}), "shapes preserved");
  ck(same_bytes(xv1, xv2) && same_bytes(xa1, xa2),
     "same seed twice -> identical bytes");
  ck(!same_bytes(xv1, g["in_video"]), "output differs from input noise");

  // slg disabled run must differ (neg pass no longer skips layer 1)
  Tensor xv3, xa3;
  make_noise(xv3, xa3, 42);
  DenoiseCfg cfg_no_slg = cfg;
  cfg_no_slg.slg_layer = -1;
  denoise(dit, cfg_no_slg, xv3, xa3, ctx_pos, ctx_neg, 42, nullptr);
  ck(!same_bytes(xv1, xv3), "slg on/off produce different latents");

  // numpy cross-check of step-0 CFG combination (pos/neg forwards + combine)
  std::string cmd = "python3 tools/golden/check_cfg_step.py " + dump +
                    " 4.0 3.0 1";
  std::printf("%s\n", cmd.c_str());
  ck(std::system(cmd.c_str()) == 0, "check_cfg_step.py cross-check");

  if (g_fails) {
    std::printf("test_denoise: %d FAILURES\n", g_fails);
    return 1;
  }
  std::printf("test_denoise: all passed\n");
  return 0;
}
