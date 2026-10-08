// End-to-end flow-matching denoise loop driving the OviDit — C++ port of
// Ovi/ovi/ovi_fusion_engine.py's sampling loop (lines 276-326): joint
// video+audio latent under two UniPC schedulers, per-branch CFG, and
// skip-layer guidance on the negative pass only.
#pragma once
#include <cstdint>
#include <functional>
#include "core/tensor.h"
#include "model/ovi_dit.h"

namespace sd {

struct DenoiseCfg {
  int steps = 50;
  double shift = 5.0;             // engine set_timesteps shift
  double video_cfg = 4.0, audio_cfg = 3.0;
  int slg_layer = 11;             // <=0 disables (fusion.py:290 `slg_layer > 0`)
};

// x_video [C,F,H,W], x_audio [L,C]: in = initial noise, out = final latent.
// text_ctx / text_ctx_neg: [2, Lc, text_dim] — row 0 is the video context,
// row 1 the audio context (engine: text_embeddings[0] pos for both branches,
// [1] video-neg, [2] audio-neg). Returns false if steps <= 0.
// Set env OVI_DENOISE_DUMP=<prefix> to dump the first 3 steps' (x_t, t,
// guided model_out) as .npy for tools/golden/check_cfg_step.py.
bool denoise(ovi::OviDit& dit, const DenoiseCfg& cfg, Tensor& x_video,
             Tensor& x_audio, const Tensor& text_ctx, const Tensor& text_ctx_neg,
             uint64_t seed, std::function<void(int, int)> progress = nullptr);

}  // namespace sd
