// Implementation of pipeline/movie.h (issue #18).
#include "pipeline/movie.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pipeline/gen.h"

namespace sd {
namespace {

struct Scene {
  std::string prompt;
  double seconds = 4.0;
  uint64_t seed = 0;
};

// ponytail: hand-rolled scanner for a flat [{string,double,int}, ...] array —
// no nested objects/arrays in the format; add a real JSON lib if scenes ever
// grow nesting.
std::vector<Scene> parse_scenes(const std::string& txt, bool& ok) {
  std::vector<Scene> out;
  ok = false;
  size_t pos = txt.find('[');
  if (pos == std::string::npos) return out;
  while (true) {
    size_t ob = txt.find('{', pos);
    if (ob == std::string::npos) break;
    // find matching '}' outside strings
    size_t i = ob + 1;
    bool in_str = false;
    for (; i < txt.size(); i++) {
      if (in_str) {
        if (txt[i] == '\\') i++;
        else if (txt[i] == '"') in_str = false;
      } else if (txt[i] == '"') in_str = true;
      else if (txt[i] == '}') break;
    }
    if (i >= txt.size()) return out;  // unbalanced
    const std::string obj = txt.substr(ob, i - ob);
    pos = i;

    auto find_key = [&](const char* key) -> size_t {
      const std::string pat = std::string("\"") + key + "\"";
      const size_t k = obj.find(pat);
      if (k == std::string::npos) return std::string::npos;
      const size_t colon = obj.find(':', k + pat.size());
      return colon == std::string::npos ? std::string::npos : colon + 1;
    };

    Scene sc;
    const size_t pk = find_key("prompt");
    if (pk == std::string::npos) return out;  // prompt is mandatory
    const size_t q1 = obj.find('"', pk);
    if (q1 == std::string::npos) return out;
    for (size_t j = q1 + 1; j < obj.size(); j++) {
      if (obj[j] == '\\') {  // keep escapes minimal: \" and \\ only
        if (j + 1 < obj.size() && (obj[j + 1] == '"' || obj[j + 1] == '\\'))
          sc.prompt += obj[++j];
        else sc.prompt += obj[j];
      } else if (obj[j] == '"') break;
      else sc.prompt += obj[j];
    }
    if (size_t sk = find_key("seconds"); sk != std::string::npos)
      sc.seconds = std::atof(obj.c_str() + sk);
    if (size_t gk = find_key("seed"); gk != std::string::npos)
      sc.seed = (uint64_t)std::strtoull(obj.c_str() + gk, nullptr, 10);
    out.push_back(sc);
  }
  ok = !out.empty();
  return out;
}

int run(const std::string& args) {
  // mux.cpp pattern: plain ffmpeg first, $HOME/.local/bin fallback
  const std::string home = getenv("HOME") ? getenv("HOME") : "";
  for (const std::string& bin :
       {std::string("ffmpeg"), home + "/.local/bin/ffmpeg"}) {
    FILE* p = popen((bin + " -hide_banner -loglevel error -y " + args).c_str(),
                    "r");
    if (!p) continue;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, p)) {}
    const int rc = pclose(p);
    if (rc != 127 && rc != -1) return rc;  // 127: shell didn't find the bin
  }
  return -1;
}

}  // namespace

int make_movie(const std::string& scenes_json_path, const std::string& out_mp4,
               int w, int h, int steps) {
  FILE* f = std::fopen(scenes_json_path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "make_movie: cannot open %s\n", scenes_json_path.c_str());
    return 1;
  }
  std::string txt;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
  std::fclose(f);

  bool ok = false;
  const std::vector<Scene> scenes = parse_scenes(txt, ok);
  if (!ok) {
    std::fprintf(stderr, "make_movie: no scenes parsed from %s\n",
                 scenes_json_path.c_str());
    return 1;
  }

  // per-scene generation (same staged pipeline as the CLI)
  std::vector<std::string> files;
  std::vector<double> durations;
  for (size_t i = 0; i < scenes.size(); i++) {
    GenCfg c;
    c.prompt = scenes[i].prompt;
    c.w = w;
    c.h = h;
    int fr = (int)(scenes[i].seconds * 24 + 0.5);
    fr = 4 * ((fr - 1) / 4) + 1;  // snap to 4k+1
    if (fr < 5) fr = 5;
    c.frames = fr;
    c.steps = steps;
    c.seed = scenes[i].seed ? scenes[i].seed : (uint64_t)i;
    c.out = out_mp4 + ".scene" + std::to_string(i) + ".mp4";
    std::printf("\n======== scene %zu/%zu (%.2fs, seed %llu) ========\n",
                i + 1, scenes.size(), (double)fr / 24.0,
                (unsigned long long)c.seed);
    std::string err;
    if (!generate(c, &err)) {
      std::fprintf(stderr, "make_movie: scene %zu failed: %s\n", i, err.c_str());
      return 1;
    }
    files.push_back(c.out);
    durations.push_back((double)fr / 24.0);
  }

  // single-scene movie: just move it into place
  if (files.size() == 1) {
    if (std::rename(files[0].c_str(), out_mp4.c_str()) != 0) return 1;
    std::printf("make_movie: wrote %s (1 scene)\n", out_mp4.c_str());
    return 0;
  }

  // concat with 0.5s crossfades: chained xfade (video) + acrossfade (audio).
  // offset_k = sum(d_0..d_{k-1}) - 0.5*k — xfade offsets are relative to the
  // accumulated first input.
  std::string cmd;  // inputs appended below, then filter_complex
  for (const auto& p : files) cmd += " -i \"" + p + "\"";
  std::string vf, af, vlabel = "0:v", alabel = "0:a";
  double total = 0;
  for (size_t i = 1; i < files.size(); i++) {
    total += durations[i - 1];
    const double offset = total - 0.5 * (double)i;
    char off[32];
    std::snprintf(off, sizeof off, "%.3f", offset);
    const std::string vn = "v" + std::to_string(i), an = "a" + std::to_string(i);
    vf += "[" + vlabel + "][" + std::to_string(i) +
          ":v]xfade=transition=fade:duration=0.5:offset=" + off + "[" + vn +
          "];";
    af += "[" + alabel + "][" + std::to_string(i) + ":a]acrossfade=d=0.5[" +
          an + "];";
    vlabel = vn;
    alabel = an;
  }
  vf.pop_back();  // trailing ';' from the loop
  af.pop_back();
  cmd += " -filter_complex \"" + vf + ";" + af + "\" -map \"[" + vlabel +
         "\"] -map \"[" + alabel + "]\" -c:v libx264 -preset veryfast -crf 20 "
         "-c:a aac \"" + out_mp4 + "\"";
  const int rc = run(cmd);
  for (const auto& p : files) std::remove(p.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "make_movie: ffmpeg xfade failed (rc=%d)\n", rc);
    return 1;
  }
  std::printf("make_movie: wrote %s (%zu scenes, 0.5s crossfades)\n",
              out_mp4.c_str(), files.size());
  return 0;
}

}  // namespace sd
