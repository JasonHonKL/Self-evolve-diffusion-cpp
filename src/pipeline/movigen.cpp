// movigen CLI (issue #16): prompt -> mp4 (video+audio) on CPU.
// Also hosts the --movie driver (issue #18).
// main() is weak: test binaries link the same objects and define their own.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "pipeline/gen.h"
#include "pipeline/movie.h"

namespace {
int usage() {
  std::printf(
      "movigen — text to video+audio, CPU only\n"
      "usage: movigen --prompt \"...\" [--out out.mp4] [--w 256] [--h 256]\n"
      "  [--frames 9] (4k+1) [--steps 6] [--seed 0] [--neg-video \"...\"]\n"
      "  [--neg-audio \"...\"] [--no-audio] [--va-chunk 2]\n"
      "  --movie scenes.json — issue #18 driver: per-scene generation + 0.5s\n"
      "    crossfade concat into one mp4\n");
  return 2;
}
}  // namespace

__attribute__((weak)) int main(int argc, char** argv) {
  sd::GenCfg c;
  std::string movie_json;
  bool have_prompt = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--prompt") { c.prompt = next("--prompt"); have_prompt = true; }
    else if (a == "--out") c.out = next("--out");
    else if (a == "--w") c.w = std::atoi(next("--w").c_str());
    else if (a == "--h") c.h = std::atoi(next("--h").c_str());
    else if (a == "--frames") c.frames = std::atoi(next("--frames").c_str());
    else if (a == "--steps") c.steps = std::atoi(next("--steps").c_str());
    else if (a == "--seed") c.seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
    else if (a == "--neg-video") c.neg_video = next("--neg-video");
    else if (a == "--neg-audio") c.neg_audio = next("--neg-audio");
    else if (a == "--no-audio") c.with_audio = false;
    else if (a == "--va-chunk") c.va_chunk = std::atoi(next("--va-chunk").c_str());
    else if (a == "--movie") movie_json = next("--movie");
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return usage(); }
  }

  if (!movie_json.empty())
    return sd::make_movie(movie_json, c.out, c.w, c.h, c.steps);
  if (!have_prompt) return usage();

  std::string err;
  return sd::generate(c, &err) ? 0 : 1;
}
