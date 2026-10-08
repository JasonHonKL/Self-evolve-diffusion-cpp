// mux.h test (issue #13): 1s sine -> mux onto a 1s black mp4; assert audio stream.
#include "pipeline/mux.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace sd;

static int run(const std::string& cmd) {
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return -1;
  char buf[4096];
  while (fgets(buf, sizeof buf, p)) {}
  return pclose(p);
}

int main() {
  const std::string dir = "/tmp/opencode/voc_mux_test";
  run("mkdir -p " + dir);
  const std::string video = dir + "/black.mp4";
  const std::string out = dir + "/out.mp4";

  // 1s black 64x64 mp4 (fallback to $HOME/.local/bin/ffmpeg if not on PATH)
  int rc = run("ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=black:s=64x64:d=1:r=10 "
               "-pix_fmt yuv420p \"" + video + "\"");
  if (rc != 0)
    rc = run(std::string(getenv("HOME") ? getenv("HOME") : "") +
             "/.local/bin/ffmpeg -hide_banner -loglevel error -y -f lavfi -i color=c=black:s=64x64:d=1:r=10 "
             "-pix_fmt yuv420p \"" + video + "\"");
  if (rc != 0) { printf("MUX TEST FAILED (could not create test video)\n"); return 1; }

  // 1s stereo 440 Hz sine, f32
  const int sr = 44100;
  const int64_t frames = sr;
  std::vector<float> pcm(frames * 2);
  for (int64_t i = 0; i < frames; i++) {
    const float v = 0.5f * std::sin(2.f * M_PI * 440.f * i / sr);
    pcm[2 * i] = v;
    pcm[2 * i + 1] = v;
  }

  if (!mux::mux_audio(video, pcm.data(), frames, 2, sr, out)) {
    printf("MUX TEST FAILED (mux_audio returned false)\n");
    return 1;
  }

  // ffprobe: audio stream present?
  std::string probe = "ffprobe -v error -select_streams a -show_entries stream=codec_name "
                      "-of csv=p=0 \"" + out + "\" 2>&1";
  FILE* p = popen(probe.c_str(), "r");
  char line[256] = {0};
  bool has_audio = false;
  while (fgets(line, sizeof line, p)) {
    if (line[0] != '\n' && line[0] != 0) has_audio = true;
  }
  pclose(p);

  printf("output audio stream: %s\n", has_audio ? line : "NONE");
  if (!has_audio) { printf("MUX TEST FAILED\n"); return 1; }
  printf("MUX TEST PASSED\n");
  return 0;
}
