#include "pipeline/mux.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace sd {
namespace mux {

static void put_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }

bool write_wav(const std::string& path, const float* pcm, int64_t frames,
               int channels, int sample_rate) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  const uint32_t data_bytes = (uint32_t)(frames * channels * 4);
  const uint32_t byte_rate = (uint32_t)(sample_rate * channels * 4);
  fwrite("RIFF", 1, 4, f);
  put_u32(f, 36 + data_bytes);
  fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f);
  put_u32(f, 16);            // fmt chunk size
  put_u16(f, 3);             // IEEE float
  put_u16(f, (uint16_t)channels);
  put_u32(f, (uint32_t)sample_rate);
  put_u32(f, byte_rate);
  put_u16(f, (uint16_t)(channels * 4));  // block align
  put_u16(f, 32);            // bits per sample
  fwrite("data", 1, 4, f);
  put_u32(f, data_bytes);
  fwrite(pcm, 1, data_bytes, f);
  const bool ok = ferror(f) == 0;
  fclose(f);
  return ok;
}

static int run(const std::string& cmd) {
  // ponytail: popen for exit status; stderr passed through for debuggability
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return -1;
  char buf[4096];
  while (fgets(buf, sizeof buf, p)) {}
  return pclose(p);
}

static int run_ffmpeg(const std::string& args) {
  const char* home = getenv("HOME");
  int rc = run("ffmpeg -hide_banner -loglevel error -y " + args);
  if (rc == 127 || rc == -1)  // shell couldn't find ffmpeg
    rc = run(std::string(home ? home : "") + "/.local/bin/ffmpeg -hide_banner -loglevel error -y " + args);
  return rc;
}

bool mux_audio(const std::string& video_mp4, const float* pcm, int64_t frames,
               int channels, int sample_rate, const std::string& out_mp4) {
  const std::string wav = out_mp4 + ".wav";
  if (!write_wav(wav, pcm, frames, channels, sample_rate)) return false;
  const std::string cmd = "-i \"" + video_mp4 + "\" -i \"" + wav +
                          "\" -c:v copy -c:a aac -shortest \"" + out_mp4 + "\"";
  const int rc = run_ffmpeg(cmd);
  remove(wav.c_str());
  return rc == 0;
}

}  // namespace mux
}  // namespace sd
