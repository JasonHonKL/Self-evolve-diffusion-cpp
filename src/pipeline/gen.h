// movigen shared generation pipeline (issues #16/#18): one prompt -> mp4
// (video+audio), 100% CPU. Staged RAM discipline (23 GB box, ~17 free):
//   T5 (f32 cache) -> encode prompts -> free_weights() -> DiT pack (12 GB)
//   -> denoise -> DiT freed -> VAE decode (chunked) -> vocoder -> mux.
#pragma once
#include <cstdint>
#include <string>

namespace sd {

struct GenCfg {
  std::string prompt;             // may contain "Audio: ..." (engine format)
  std::string out = "out.mp4";
  int w = 256, h = 256;           // pixels, multiples of 16
  int frames = 9;                 // rgb frames, must be 4k+1
  int steps = 6;
  uint64_t seed = 0;
  std::string neg_video = "jitter, bad hands, blur, distortion";
  std::string neg_audio = "robotic, muffled, echo, distorted";
  bool with_audio = true;
  int va_chunk = 2;               // latent frames per VAE decode_chunk
  int fps = 24;
};

// Latent math derived from Ovi/ovi/ovi_fusion_engine.py:
//   video: [48, F, h/16, w/16]            (engine 216-218, 262)
//   audio: [T, 20] @31.25 Hz              (engine 133-136, 263)
//   F = (frames-1)/4 + 1; T fits engine's (157 @5s, 314 @10s) i.e.
//   T = floor((frames-1)/24 * 31.25 + 0.5) + 1.
int video_latent_frames(int rgb_frames);
int audio_latent_len(int rgb_frames);

// Runs the full pipeline; prints per-step progress with ETA, free RAM per
// stage, and a stage timing table. Returns false + err on failure.
bool generate(const GenCfg& cfg, std::string* err = nullptr);

}  // namespace sd
