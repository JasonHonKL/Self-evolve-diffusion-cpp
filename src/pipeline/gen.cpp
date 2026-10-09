// Implementation of pipeline/gen.h — the movigen core (issues #16/#18).
// Component usage mirrors the reference engine (Ovi/ovi/ovi_fusion_engine.py):
//   text: 3 prompts (pos, video-neg, audio-neg), T5 contexts truncated to
//         their own token length (t5.py:515-518), zero-padded to a common Lc
//         (model.py:784-789 padded rows ARE attended);
//   noise: randn with the same seed for both branches (engine 262-263);
//   guidance: engine defaults video_cfg=5.0, audio_cfg=4.0, slg=9, shift=5
//         (engine 150-154);
//   post: VAE out [3,T,H,W] in [-1,1] -> (x+1)/2*255 uint8 rgb
//         (io_utils.py:44-46); audio wav mono 16 kHz (vocoder contract).
#include "pipeline/gen.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>
#include <utility>
#include <vector>

#include "model/ovi_dit.h"
#include "model/t5enc.h"
#include "model/vae_dec.h"
#include "model/vocoder.h"
#include "pipeline/denoise.h"
#include "pipeline/mux.h"
#include "tokenizer/spm.h"

namespace sd {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

// ---- /proc/meminfo helper (stage RAM logging) ----
double mem_avail_gb() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[128];
  long kb = -1;
  while (std::fgets(line, sizeof line, f))
    if (!std::strncmp(line, "MemAvailable:", 13)) {
      kb = std::atol(line + 13);
      break;
    }
  std::fclose(f);
  return kb / 1048576.0;  // kB -> GB
}

struct StageLog {
  std::vector<std::pair<std::string, double>> v;
  std::string cur;
  Clock::time_point t0;
  void begin(const std::string& n) {
    finish();
    cur = n;
    t0 = Clock::now();
    std::printf("\n== %-28s [mem avail %.1f GB]\n", n.c_str(), mem_avail_gb());
    std::fflush(stdout);
  }
  void finish() {
    if (!cur.empty()) {
      v.emplace_back(cur, since(t0));
      std::printf("== %-28s done in %6.1f s\n", cur.c_str(), v.back().second);
      cur.clear();
    }
  }
  void table() const {
    double total = 0;
    std::printf("\n---- stage timing table ----\n");
    for (const auto& s : v) {
      std::printf("%-28s %8.1f s\n", s.first.c_str(), s.second);
      total += s.second;
    }
    std::printf("%-28s %8.1f s\n", "TOTAL", total);
    std::fflush(stdout);
  }
};

// ---- gaussian noise: splitmix64 + Box-Muller (engine: torch.randn, 262-263).
// core only has uniform randu; gaussian lives here per the ownership rules.
void randn(Tensor& x, uint64_t seed) {
  uint64_t s = seed + 0x9E3779B97F4A7C15ull;
  auto next = [&s]() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  float* p = x.ptr<float>();
  const int64_t n = x.numel();
  for (int64_t i = 0; i < n; i += 2) {
    const double u1 = ((next() >> 11) + 1.0) / 9007199254740993.0;  // (0,1]
    const double u2 = (next() >> 11) / 9007199254740992.0;          // [0,1)
    const double r = std::sqrt(-2.0 * std::log(u1));
    p[i] = (float)(r * std::cos(6.283185307179586 * u2));
    if (i + 1 < n) p[i + 1] = (float)(r * std::sin(6.283185307179586 * u2));
  }
}

// ---- ffmpeg/ffprobe popen helpers (mux.cpp's pattern, write-direction) ----
std::string ffmpeg_bin() {
  for (const std::string& b : {std::string("ffmpeg"),
                               std::string(getenv("HOME") ? getenv("HOME") : "") +
                                   "/.local/bin/ffmpeg"}) {
    const std::string probe = b + " -version 2>/dev/null";
    FILE* p = popen(probe.c_str(), "r");
    if (!p) continue;
    char buf[256];
    const bool ok = std::fgets(buf, sizeof buf, p) != nullptr;
    if (pclose(p) == 0 && ok) return b;
  }
  return "ffmpeg";
}

FILE* ffmpeg_pipe_write(const std::string& args) {
  return popen((ffmpeg_bin() + " -hide_banner -loglevel error -y " + args)
                   .c_str(),
               "w");
}

// [3,T,H,W] chunk f32 [-1,1] -> one rgb24 frame into the ffmpeg stdin.
void write_rgb_frame(FILE* f, const Tensor& out, int64_t t, int H, int W) {
  const float* p = out.ptr<float>();
  const int64_t plane = out.dim(1) * H * W, HW = (int64_t)H * W;
  std::vector<unsigned char> row(W * 3);
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      const int64_t idx = t * HW + (int64_t)h * W + w;
      for (int c = 0; c < 3; c++) {
        float v = (p[c * plane + idx] + 1.f) * 127.5f;
        row[w * 3 + c] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
      }
    }
    if (std::fwrite(row.data(), 3, W, f) != (size_t)W) return;
  }
}

// [48,F,H,W] -> non-owning [48,Tc,H,W] slice of latent frames [t0,t1).
Tensor slice_latent(const Tensor& z, int64_t t0, int64_t t1) {
  const int64_t F = z.dim(1), H = z.dim(2), W = z.dim(3);
  Tensor c({48, t1 - t0, H, W}, DType::F32);
  for (int64_t ch = 0; ch < 48; ch++)
    std::memcpy(c.ptr<float>() + ch * (t1 - t0) * H * W,
                z.ptr<float>() + ch * F * H * W + t0 * H * W,
                (size_t)(t1 - t0) * H * W * 4);
  return c;
}

// encode one prompt -> [L,4096] T5 context (L = own token count, <=512).
Tensor encode_prompt(t5::T5Encoder& enc, const spm::SpmModel& spm,
                     const std::string& text) {
  std::vector<int> ids = spm.encode(text, /*add_eos=*/true);
  if ((int)ids.size() > 512) ids.resize(512);  // engine tokenizer seq_len=512
  const int64_t L = (int64_t)ids.size();
  Tensor id({L}, DType::F32), mask({L}, DType::F32), out;
  for (int64_t i = 0; i < L; i++) {
    id.ptr<float>()[i] = (float)ids[i];
    mask.ptr<float>()[i] = 1.f;
  }
  enc.forward(id, mask, out);
  return out;
}

// [L,4096] + [L,4096] -> zero-padded stack [2, Lc, 4096] (denoise contract).
Tensor stack_ctx(const Tensor& a, const Tensor& b, int64_t Lc) {
  const int64_t D = a.dim(1);
  Tensor t({2, Lc, D}, DType::F32);  // ctor zeroes -> pad rows are zeros
  std::memcpy(t.ptr<float>(), a.ptr<float>(), a.nbytes);
  std::memcpy((char*)t.ptr<float>() + a.nbytes, b.ptr<float>(), b.nbytes);
  return t;
}

}  // namespace

int video_latent_frames(int rgb_frames) { return (rgb_frames - 1) / 4 + 1; }

int audio_latent_len(int rgb_frames) {
  const double s = (double)(rgb_frames - 1) / 24.0;
  return (int)std::floor(s * 31.25 + 0.5) + 1;
}

bool generate(const GenCfg& c, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    std::fprintf(stderr, "\n[movigen] FAILED: %s\n", m.c_str());
    return false;
  };

  if (c.frames < 5 || (c.frames - 1) % 4 != 0)
    return fail("--frames must be 4k+1 (5, 9, 13, ...)");
  if (c.w % 16 || c.h % 16 || c.w < 16 || c.h < 16)
    return fail("--w/--h must be positive multiples of 16");
  if (c.steps < 1) return fail("--steps must be >= 1");
  if (c.prompt.empty()) return fail("--prompt is required");

  const int F = video_latent_frames(c.frames), Hl = c.h / 16, Wl = c.w / 16;
  const int T = audio_latent_len(c.frames);
  std::printf(
      "movigen: %dx%d %d frames (%.2fs) %d steps seed %llu\n"
      "  video latent [48,%d,%d,%d] (%d tokens), audio latent [%d,20] @31.25Hz\n",
      c.w, c.h, c.frames, (double)c.frames / c.fps, c.steps,
      (unsigned long long)c.seed, F, Hl, Wl, F * Hl * Wl / 4, T);

  StageLog log;

  // ---- stage 1: tokenizer + T5 (encode all prompts, then free) ----
  log.begin("tokenizer");
  spm::SpmModel spm;
  if (!spm.load("ckpts/umt5_tokenizer.model") &&
      !spm.load("ckpts/umt5_tokenizer/spiece.model"))
    return fail("tokenizer load failed");
  log.finish();

  log.begin("T5 encode (f32 cache)");
  Tensor ctx_pos, ctx_neg;  // [2, Lc, 4096] each, alive through denoise
  {
    t5::T5Encoder enc("ckpts/t5_enc.safetensors");
    enc.f32_cache = true;
    Tensor pos = encode_prompt(enc, spm, c.prompt);
    Tensor vneg = encode_prompt(enc, spm, c.neg_video);
    Tensor aneg = encode_prompt(enc, spm, c.neg_audio);
    const int64_t Lc = std::max({pos.dim(0), vneg.dim(0), aneg.dim(0)});
    ctx_pos = stack_ctx(pos, pos, Lc);        // row0 video-ctx, row1 audio-ctx
    ctx_neg = stack_ctx(vneg, aneg, Lc);      // engine 244-248
    log.finish();
    log.begin("T5 free_weights");
    enc.free_weights();  // ~19 GB back before the DiT pack loads
    malloc_trim(0);
    log.finish();
  }

  // ---- stage 2: DiT pack + denoise ----
  Tensor xv, xa;
  {
    log.begin("DiT pack load (12GB)");
    ovi::OviDit dit;
    dit.init(ovi::OviDitCfg::real());
    dit.load_quantized_pack("ckpts/Ovi/model.sdcpp");
    log.finish();

    log.begin("denoise");
    xv = Tensor({48, F, Hl, Wl}, DType::F32);
    xa = Tensor({T, 20}, DType::F32);
    randn(xv, c.seed);  // engine 262-263: fresh generator per branch,
    randn(xa, c.seed);  // same seed -> our two fills share it too
    DenoiseCfg d;
    d.steps = c.steps;
    d.shift = 5.0;      // engine 151
    d.video_cfg = 5.0;  // engine 152
    d.audio_cfg = 4.0;  // engine 153
    d.slg_layer = 9;    // engine 154
    const auto tden = Clock::now();
    auto progress = [&](int i, int n) {
      const double el = since(tden);
      const double eta = i > 0 ? el / i * (n - i) : 0;
      std::printf("\r  step %d/%d  elapsed %5.0fs  eta %5.0fs", i, n, el, eta);
      std::fflush(stdout);
    };
    if (!denoise(dit, d, xv, xa, ctx_pos, ctx_neg, c.seed, progress)) {
      return fail("denoise returned false");
    }
    std::printf("\n");
    log.finish();
  }  // OviDit destroyed here: 12 GB pack released before VAE/vocoder
  malloc_trim(0);

  // free text ctx as well before VAE activations
  ctx_pos = Tensor();
  ctx_neg = Tensor();

  // ---- stage 3: VAE decode (chunked) -> x264 via ffmpeg rawvideo pipe ----
  const std::string tmp_video = c.out + ".video.mp4";
  {
    log.begin("VAE decode + x264");
    vae::VaeDecoder vae("ckpts/vae_dec.safetensors");
    FILE* ff = ffmpeg_pipe_write(
        "-f rawvideo -pix_fmt rgb24 -s " + std::to_string(c.w) + "x" +
        std::to_string(c.h) + " -r " + std::to_string(c.fps) +
        " -i - -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \"" +
        tmp_video + "\"");
    if (!ff) return fail("popen(ffmpeg) failed");
    vae::VaeDecoder::Cache cache;
    int64_t written = 0;
    for (int64_t t0 = 0; t0 < F; t0 += c.va_chunk) {
      const int64_t t1 = std::min<int64_t>(F, t0 + c.va_chunk);
      Tensor chunk, out;
      Tensor z = slice_latent(xv, t0, t1);
      vae.decode_chunk(z, out, cache);  // anchor chunk emits 4Tc-3 frames
      const int64_t Tf = out.dim(1);
      for (int64_t t = 0; t < Tf; t++)
        write_rgb_frame(ff, out, t, c.h, c.w);
      written += Tf;
    }
    const int rc = pclose(ff);
    if (rc != 0) return fail("ffmpeg video encode failed (rc=" +
                             std::to_string(rc) + ")");
    if (written != c.frames)
      return fail("VAE emitted " + std::to_string(written) + " frames, want " +
                  std::to_string(c.frames));
    log.finish();
  }

  // ---- stage 4: vocoder + mux (audio), or plain video out ----
  if (c.with_audio) {
    log.begin("vocoder");
    Tensor z({20, T}, DType::F32), wav;
    for (int t = 0; t < T; t++)          // denoise audio latent [T,20] ->
      for (int ch = 0; ch < 20; ch++)    // vocoder z [20,T] (engine 339)
        z.ptr<float>()[ch * T + t] = xa.ptr<float>()[t * 20 + ch];
    voc::Vocoder voc("ckpts/mmaudio_vocoder.safetensors");
    voc.latent_to_wav(z, wav);  // [1, 512T] mono 16 kHz, in (-1,1)
    log.finish();
    log.begin("mux audio");
    // pad audio with silence past the video duration so the mux's -shortest
    // can't trim video frames (aac packs 1024-sample packets; a video-length
    // wav can end an aac packet early and chop the last frame)
    std::vector<float> pcm(wav.ptr<float>(),
                           wav.ptr<float>() + wav.numel());
    const double vsamples = (double)c.frames / c.fps * 16000.0;
    const int64_t want =
        (int64_t)std::ceil(vsamples / 1024.0) * 1024 + 2048;
    if ((int64_t)pcm.size() < want) pcm.resize(want, 0.f);
    const bool ok =
        mux::mux_audio(tmp_video, pcm.data(), (int64_t)pcm.size(), 1, 16000,
                       c.out);
    if (!ok) return fail("mux_audio failed");
    std::remove(tmp_video.c_str());
    log.finish();
  } else {
    if (std::rename(tmp_video.c_str(), c.out.c_str()) != 0)
      return fail("rename to output failed");
  }

  log.finish();
  log.table();
  FILE* sf = std::fopen(c.out.c_str(), "rb");
  long sz = 0;
  if (sf) {
    std::fseek(sf, 0, SEEK_END);
    sz = std::ftell(sf);
    std::fclose(sf);
  }
  std::printf("movigen: wrote %s (%.1f KB)\n", c.out.c_str(), sz / 1024.0);
  return true;
}

}  // namespace sd
