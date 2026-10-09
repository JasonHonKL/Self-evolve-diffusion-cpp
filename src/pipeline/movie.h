// make_movie (issue #18): scenes JSON -> per-scene movigen runs -> single mp4
// joined with 0.5 s ffmpeg xfade/acrossfade crossfades.
#pragma once
#include <string>

namespace sd {

// scenes JSON: [{"prompt": "...", "seconds": 4.0, "seed": 123}, ...]
// (seed optional, defaults to scene index; seconds rounded to a 4k+1 frame
// count). All scenes share w/h/steps. Returns 0 on success, nonzero else.
int make_movie(const std::string& scenes_json_path, const std::string& out_mp4,
               int w = 256, int h = 256, int steps = 6);

}  // namespace sd
