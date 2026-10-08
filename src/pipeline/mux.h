// Minimal WAV writer + ffmpeg audio muxer (issue #13).
#pragma once
#include <cstdint>
#include <string>

namespace sd {
namespace mux {

// pcm: interleaved f32 frames (frames*channels floats)
bool write_wav(const std::string& path, const float* pcm, int64_t frames,
               int channels, int sample_rate);

// Writes wav to "<out>.wav", then ffmpeg -i video -i wav -c:v copy -c:a aac
// -shortest out. Temp wav removed on success.
bool mux_audio(const std::string& video_mp4, const float* pcm, int64_t frames,
               int channels, int sample_rate, const std::string& out_mp4);

}  // namespace mux
}  // namespace sd
