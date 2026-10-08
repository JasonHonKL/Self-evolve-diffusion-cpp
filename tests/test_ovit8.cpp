// issue #17: int8 DiT.
//  (1) PARITY: scaled-down golden weights -> tiny safetensors -> pack_sdcpp.py
//      -> load_quantized_pack -> forward, vs the fp32 forward: cosine > 0.99.
//  (2) REAL BENCH: ckpts/Ovi/model.sdcpp (12GB), one real-config forward at
//      small token counts (video [48,4,8,8] -> 64 tokens, audio [100,20],
//      text 64), 3 iterations, lin-vs-attn section split. fp32 real bench
//      skipped: fp32 load() would materialize ~46GB of weights (23GB RAM).
//  (3) PROJECT: scale measured ms by video-token ratio to 720p 5s.
//  (4) T5: encode s/prompt, bf16-convert vs fp32-cached (exercises the
//      issue #17 cache + free_weights; runs BEFORE the DiT bench for RAM).
#include "model/ovi_dit.h"
#include "model/t5enc.h"
#include "serde/safetensors.h"
#include "test_support/npz.h"
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace sd;

static int g_fails = 0;
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fails++; }                  \
    else { printf("ok  : %s\n", msg); }                                     \
  } while (0)

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double cosine(const Tensor& a, const Tensor& b) {
  int64_t n = a.numel();
  const float* pa = a.ptr<float>();
  const float* pb = b.ptr<float>();
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; i++) {
    dot += (double)pa[i] * pb[i];
    na += (double)pa[i] * pa[i];
    nb += (double)pb[i] * pb[i];
  }
  return dot / (std::sqrt(na * nb) + 1e-30);
}

static std::string repo_root() {
  std::string f = __FILE__;
  size_t p = f.rfind("/tests/");
  return p == std::string::npos ? "." : f.substr(0, p);
}

static bool run(const std::string& cmd) {
  printf("-- %s\n", cmd.c_str());
  return system(cmd.c_str()) == 0;
}

static bool file_ok(const char* p, int64_t min_bytes) {
  struct stat st;
  return stat(p, &st) == 0 && st.st_size >= min_bytes;
}

// ponytail: shared box — other agents' jobs eat RAM; big parts self-skip
// instead of racing the OOM killer (T5 fp32 cache needs ~15GB, DiT ~13GB).
static double avail_gb() {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char k[64];
  double v = 0, kb;
  while (fscanf(f, "%63s %lf %*s", k, &kb) == 2)
    if (std::string(k) == "MemAvailable:") { v = kb / 1048576.0; break; }
  fclose(f);
  return v;
}

// ---- (1) int8-pack parity vs fp32 forward on the golden config ------------
static void part_parity() {
  printf("\n== part 1: int8-pack parity (golden cfg) ==\n");
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/ovit_golden.npz");
  } catch (const std::exception&) {
    if (!run("python3 " + repo_root() + "/tools/golden/gen_ovit.py")) exit(1);
    g = load_npz("tools/golden/ovit_golden.npz");
  }
  // pack the whole golden map (extra non-weight arrays are <65536 -> raw f32,
  // ignored by the loader; the packer quantizes every Linear >=65536)
  write_safetensors("/tmp/opencode/ovit8_small.st", g);
  if (!run("python3 " + repo_root() + "/tools/quantize/pack_sdcpp.py"
           " /tmp/opencode/ovit8_small.st /tmp/opencode/ovit8_small.sdcpp"))
    exit(1);

  ovi::OviDit fp32;
  fp32.init(ovi::OviDitCfg::test());
  fp32.load(g);
  ovi::OviDit i8;
  i8.init(ovi::OviDitCfg::test());
  i8.load_quantized_pack("/tmp/opencode/ovit8_small.sdcpp");
  CHECK(i8.use_int8, "load_quantized_pack sets use_int8");

  float t = g["t"].ptr<float>()[0];
  Tensor ovf, oaf, ov8, oa8;
  fp32.forward(g["in_video"], g["in_audio"], t, g["in_vctx"], g["in_actx"], ovf, oaf);
  i8.forward(g["in_video"], g["in_audio"], t, g["in_vctx"], g["in_actx"], ov8, oa8);
  char msg[96];
  double cv = cosine(ov8, ovf), ca = cosine(oa8, oaf);
  snprintf(msg, sizeof msg, "int8-vs-fp32 out_video cosine=%.4f (>0.99)", cv);
  CHECK(cv > 0.99, msg);
  snprintf(msg, sizeof msg, "int8-vs-fp32 out_audio cosine=%.4f (>0.99)", ca);
  CHECK(ca > 0.99, msg);
  snprintf(msg, sizeof msg, "int8-vs-golden out_video cosine=%.4f (>0.99)",
           cosine(ov8, g["out_video"]));
  CHECK(cosine(ov8, g["out_video"]) > 0.99, msg);
  snprintf(msg, sizeof msg, "int8-vs-golden out_audio cosine=%.4f (>0.99)",
           cosine(oa8, g["out_audio"]));
  CHECK(cosine(oa8, g["out_audio"]) > 0.99, msg);
  unlink("/tmp/opencode/ovit8_small.st");
  unlink("/tmp/opencode/ovit8_small.sdcpp");
}

// ---- (4) T5 fp32 weight cache (issue #17): golden parity + s/prompt -------
// ponytail: the uncached (per-call bf16->f32) path is already timed by
// test_t5enc (60-84 s/prompt at L=16); here we verify the CACHED path against
// the same golden and time it at L=16 and L=512. T5 runs before the DiT so
// the 14.5GB fp32 cache + 12GB DiT pack never coexist (free_weights after).
static void part_t5() {
  printf("\n== part 4: T5 encode — fp32-cached (golden parity + timing) ==\n");
  const char* ck = "ckpts/t5_enc.safetensors";
  if (!file_ok(ck, 9000000000ll)) {
    printf("skip: %s absent/incomplete\n", ck);
    return;
  }
  if (avail_gb() < 17.5) {
    printf("skip: only %.1f GB available (fp32 cache needs ~18.5 GB)\n",
           avail_gb());
    return;
  }
  std::unordered_map<std::string, Tensor> g;
  try {
    g = load_npz("tools/golden/t5_golden.npz");
  } catch (const std::exception& e) {
    printf("skip golden parity (no t5_golden.npz: %s)\n", e.what());
  }
  t5::T5Encoder enc(ck);
  const int64_t D = t5::T5Encoder::dim;

  if (!g.empty()) {
    const Tensor& ids = g.at("ids");
    const Tensor& mask = g.at("mask");
    const Tensor& hidden = g.at("hidden");
    int64_t B = ids.dim(0), L = ids.dim(1);
    enc.f32_cache = true;
    double worst = 1;
    for (int64_t b = 0; b < B; b++) {
      Tensor id1({L}, DType::F32), m1({L}, DType::F32), out;
      memcpy(id1.data, ids.ptr<float>() + b * L, (size_t)L * 4);
      memcpy(m1.data, mask.ptr<float>() + b * L, (size_t)L * 4);
      double t0 = now_ms();
      enc.forward(id1, m1, out);
      double ms = now_ms() - t0;
      const float* o = out.ptr<float>();
      const float* r = hidden.ptr<float>() + b * L * D;
      double dot = 0, no = 0, nr = 0;
      for (int64_t i = 0; i < L * D; i++) {
        dot += (double)o[i] * r[i];
        no += (double)o[i] * o[i];
        nr += (double)r[i] * r[i];
      }
      double cos = dot / (std::sqrt(no) * std::sqrt(nr) + 1e-30);
      worst = std::min(worst, cos);
      printf("cached prompt %lld (L=%lld): cosine=%.6f [%.0f ms%s]\n",
             (long long)b, (long long)L, cos, ms, b == 0 ? ", incl cache build]" : "");
    }
    char msg[96];
    snprintf(msg, sizeof msg, "cached golden cosine worst=%.6f (>0.98)", worst);
    CHECK(worst > 0.98, msg);
  } else {
    enc.f32_cache = true;
  }

  Tensor ids({512}, DType::F32), mask({512}, DType::F32), o1, o2;
  randu(ids, 23);
  for (int64_t i = 0; i < 512; i++) {
    ids.ptr<float>()[i] = (float)(std::abs((int)ids.ptr<float>()[i]) % 200000);
    mask.ptr<float>()[i] = 1.f;
  }
  enc.forward(ids, mask, o1);  // warm at realistic length
  double t0 = now_ms();
  enc.forward(ids, mask, o2);
  double ms512 = now_ms() - t0;
  printf("T5 s/prompt (cached, fp32 weights): L=512 %.2f s\n", ms512 / 1e3);
  enc.free_weights();  // pipeline order: T5 first, then DiT (23GB RAM)
  printf("T5 weights freed (fp32 cache + mmap)\n");
}

// ---- (2)+(3) real-config int8 bench + token-scaling projection -----------
static void part_real_bench() {
  printf("\n== part 2: REAL int8 bench (one DiT forward) ==\n");
  const char* ck = "ckpts/Ovi/model.sdcpp";
  if (!file_ok(ck, 11900000000ll)) {
    printf("skip: %s absent/incomplete\n", ck);
    return;
  }
  if (avail_gb() < 13) {
    printf("skip: only %.1f GB available (int8 bench touches ~12 GB pack)\n",
           avail_gb());
    return;
  }
  // bench inputs: video latent [48,4,8,8] -> F=4, Hp=Wp=8/2=4 -> 64 tokens
  const int64_t F = 4, H = 8, W = 8;
  ovi::OviDit m;
  m.init(ovi::OviDitCfg::real());
  double t0 = now_ms();
  m.load_quantized_pack(ck);
  printf("pack load (mmap, lazy): %.0f ms\n", now_ms() - t0);
  CHECK(m.use_int8, "real pack sets use_int8");

  Tensor nv({48, F, H, W}, DType::F32), na({100, 20}, DType::F32);
  Tensor vc({64, 4096}, DType::F32), ac({64, 4096}, DType::F32);
  randu(nv, 3); randu(na, 5); randu(vc, 7); randu(ac, 9);
  Tensor ov, oa;

  const int64_t Hp = H / 2, Wp = W / 2, tok_bench = F * Hp * Wp;
  printf("tokens: video=%lld (F=%lld H/W=%lld->%lld patch(1,2,2)), audio=100, "
         "text=64->padded 512\n",
         (long long)tok_bench, (long long)F, (long long)H, (long long)Hp);

  ovi::section_timers(true);
  double best = 1e30, best_lin = 0, best_attn = 0;
  for (int it = 0; it < 3; it++) {
    t0 = now_ms();
    m.forward(nv, na, 3.0f, vc, ac, ov, oa);
    double ms = now_ms() - t0;
    double lms = ovi::section_ms_lin(), ams = ovi::section_ms_attn();
    ovi::section_timers(true);  // reset accumulators
    printf("iter %d: %8.1f ms  (linear/gemm %7.1f ms, sdpa %7.1f ms)%s\n", it,
           ms, lms, ams, it == 0 ? "  [cold: pack page-in]" : "");
    if (ms < best) { best = ms; best_lin = lms; best_attn = ams; }
  }

  // ---- (3) projection to real 720p 5s latent [48,31,45,45] ----
  // exact formula the forward uses: Lv = F * floor(H/2) * floor(W/2)
  const int64_t Fr = 31, Hr = 45, Wr = 45;
  const int64_t Hpr = Hr / 2, Wpr = Wr / 2, tok_real = Fr * Hpr * Wpr;
  const double r = (double)tok_real / (double)tok_bench;
  const double step_lin = best_lin * r;      // gemms scale ~linearly in tokens
  // sdpa bucket is quadratic for video self-attn (dominant); cross-attn
  // (Lq x 512 text) and audio self-attn only scale linearly -> upper bound.
  const double step_attn = best_attn * r * r;
  const double step = step_lin + step_attn;
  printf("\n== part 3: token-scaling projection ==\n");
  printf("tokens_bench=%lld  tokens_real=F*floor(H/2)*floor(W/2)"
         "=%lld*%lld*%lld=%lld  (ceil/padded: %lld*23*23=%lld)  r=%.1f\n",
         (long long)tok_bench, (long long)Fr, (long long)Hpr, (long long)Wpr,
         (long long)tok_real, (long long)Fr,
         (long long)Fr * ((Hr + 1) / 2) * ((Wr + 1) / 2), r);
  printf("measured @%lld tok: total %.1f ms = lin %.1f + attn %.1f + other %.1f\n",
         (long long)tok_bench, best, best_lin, best_attn,
         best - best_lin - best_attn);
  printf("projected @%lld tok: step = lin*%.1f + attn*%.1f^2 = %.0f + %.0f = "
         "%.0f ms/step\n",
         (long long)tok_real, r, r, step_lin, step_attn, step);
  printf("50-step clip: %.1f min  (x2 for CFG: %.1f min)\n", step * 50 / 6e4,
         step * 100 / 6e4);
}

int main() {
  part_parity();
  part_t5();
  part_real_bench();
  if (g_fails) { printf("test_ovit8: %d FAILURES\n", g_fails); return 1; }
  printf("test_ovit8: all passed\n");
  return 0;
}
