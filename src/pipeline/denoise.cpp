// Denoise loop — port of Ovi/ovi/ovi_fusion_engine.py lines 276-326.
// Control flow mirrored exactly:
//   - two UniPC schedulers (video/audio), same steps+shift (engine 191-200)
//   - model fed t_v for BOTH backbones (engine 277: full((1,), t_v))
//   - cond and uncond are TWO separate forwards, not one batched call
//     (engine 291-313); slg_layer only in the negative pass (engine 305)
//   - guided = neg + scale * (pos - neg), no x0 clamp (engine 316-317)
//   - each branch steps its own scheduler with its own t (engine 320-325)
#include "pipeline/denoise.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sampler/unipc.h"

namespace sd {
namespace {

// [2, Lc, D] -> non-owning view of row `r` as [Lc, D].
Tensor ctx_row(const Tensor& batch, int r) {
  Tensor v;
  v.shape = {(int64_t)batch.dim(1), (int64_t)batch.dim(2)};
  v.dtype = batch.dtype;
  v.data = static_cast<char*>(batch.data) + (size_t)r * v.numel() * 4;
  v.nbytes = (size_t)v.numel() * 4;
  v.owning = false;
  return v;
}

// engine 316-317: neg + scale * (pos - neg), elementwise.
Tensor guided(const Tensor& pos, const Tensor& neg, double scale) {
  Tensor g(pos.shape, DType::F32);
  const float* p = pos.ptr<float>();
  const float* n = neg.ptr<float>();
  float* o = g.ptr<float>();
  for (int64_t i = 0; i < pos.numel(); i++)
    o[i] = n[i] + (float)scale * (p[i] - n[i]);
  return g;
}

// Minimal .npy (f32, C order) writer for the CFG debug dump.
void save_npy(const std::string& path, const float* data, size_t n,
              const std::vector<int64_t>& shape) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::string sh = "(";
  for (auto d : shape) sh += std::to_string(d) + ",";
  sh += ")";
  char hdr[256];
  int hl = std::snprintf(hdr, sizeof hdr,
                         "{'descr': '<f4', 'fortran_order': False, "
                         "'shape': %s, }", sh.c_str());
  int pad = 64 - (10 + hl) % 64;  // numpy 64-byte alignment
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  uint16_t total = (uint16_t)(hl + pad);
  std::fwrite(&total, 2, 1, f);
  std::fwrite(hdr, 1, hl, f);
  std::fwrite(std::string(pad - 1, ' ').c_str(), 1, pad - 1, f);
  std::fputc('\n', f);
  std::fwrite(data, 4, n, f);
  std::fclose(f);
}

void save_npy(const std::string& path, const Tensor& x) {
  save_npy(path, x.ptr<float>(), (size_t)x.numel(), x.shape);
}

}  // namespace

bool denoise(ovi::OviDit& dit, const DenoiseCfg& c, Tensor& x_video,
             Tensor& x_audio, const Tensor& text_ctx, const Tensor& text_ctx_neg,
             uint64_t seed, std::function<void(int, int)> progress) {
  (void)seed;  // ponytail: caller supplies the noise; loop itself is
               // deterministic, seed kept for engine API parity
  if (c.steps <= 0) return false;

  UniPCScheduler sched_video, sched_audio;  // engine 191-200: two schedulers
  sched_video.set_timesteps(c.steps, c.shift);
  sched_audio.set_timesteps(c.steps, c.shift);

  const Tensor vid_pos = ctx_row(text_ctx, 0), aud_pos = ctx_row(text_ctx, 1);
  const Tensor vid_neg = ctx_row(text_ctx_neg, 0), aud_neg = ctx_row(text_ctx_neg, 1);

  const char* dump = std::getenv("OVI_DENOISE_DUMP");
  const int saved_slg = dit.cfg.slg_layer;
  Tensor pv, pa, nv, na, gv, ga;
  std::vector<float> tmp_v((size_t)x_video.numel()), tmp_a((size_t)x_audio.numel());

  for (int i = 0; i < c.steps; i++) {
    const int64_t t_v = sched_video.timesteps()[(size_t)i];
    const int64_t t_a = sched_audio.timesteps()[(size_t)i];

    dit.cfg.slg_layer = -1;  // positive pass: no SLG (engine 283-296)
    dit.forward(x_video, x_audio, (float)t_v, vid_pos, aud_pos, pv, pa);
    dit.cfg.slg_layer = c.slg_layer;  // negative pass carries SLG (engine 305)
    dit.forward(x_video, x_audio, (float)t_v, vid_neg, aud_neg, nv, na);

    gv = guided(pv, nv, c.video_cfg);   // engine 316
    ga = guided(pa, na, c.audio_cfg);   // engine 317

    if (dump && i < 3) {  // debug flag for tools/golden/check_cfg_step.py
      std::string p = std::string(dump) + "_s" + std::to_string(i);
      save_npy(p + "_xv.npy", x_video);
      save_npy(p + "_xa.npy", x_audio);
      save_npy(p + "_pv.npy", gv);
      save_npy(p + "_pa.npy", ga);
      float tf = (float)t_v;
      save_npy(p + "_t.npy", &tf, 1, {1});
    }

    // engine 320-325: each branch steps its own scheduler with its own t.
    sched_video.step(gv.ptr<float>(), t_v, x_video.ptr<float>(),
                     tmp_v.data(), tmp_v.size());
    std::memcpy(x_video.ptr<float>(), tmp_v.data(), tmp_v.size() * 4);
    sched_audio.step(ga.ptr<float>(), t_a, x_audio.ptr<float>(),
                     tmp_a.data(), tmp_a.size());
    std::memcpy(x_audio.ptr<float>(), tmp_a.data(), tmp_a.size() * 4);

    if (progress) progress(i + 1, c.steps);
  }
  dit.cfg.slg_layer = saved_slg;
  return true;
}

}  // namespace sd
