// MMAudio audio path port (issue #13): Ovi's audio latent -> WAV.
// Stage 1: MMAudio VAE decoder (v1-16.pth)  z[20,T]@31.25Hz -> mel[80,2T]@62.5Hz
// Stage 2: BigVGAN v1 generator (best_netG) mel[80,T]@62.5Hz -> wav[1,256T]@16kHz
// Weights: tools/convert_vocoder.py -> ckpts/mmaudio_vocoder.safetensors
// (weight_norm / MPConv normalization already folded; snake alpha/beta pre-exp'd).
#pragma once
#include "core/tensor.h"
#include "serde/safetensors.h"
#include <functional>
#include <string>
#include <vector>

namespace voc {
using sd::Tensor;

// optional debug hook: called with stage name + [C,T] snapshot when non-null
void set_trace(const std::function<void(const std::string&, const Tensor&)>& fn);

// one BigVGAN AMPBlock (resblock index ib, kernel kj) — exposed for tests
Tensor amp_block(const sd::SafetensorsFile& st, const Tensor& x, int64_t ib, int64_t kj);

class Vocoder {
public:
  explicit Vocoder(const std::string& safetensors_path);

  // z [20, T] f32 -> mel [80, 2T] (unnormalized log10-mel)
  void decode_latent(const Tensor& z, Tensor& mel) const;
  // mel [80, T] f32 -> wav [1, 256T] in (-1, 1)
  void vocode(const Tensor& mel, Tensor& wav) const;
  // full chain: z [20, T] -> wav [1, 512T]
  void latent_to_wav(const Tensor& z, Tensor& wav) const;

private:
  const sd::Tensor& W(const std::string& key) const;

  sd::SafetensorsFile st;
  std::vector<sd::Tensor> ups_w_;  // BigVGAN ups weights permuted to [out, in, k]
  float dec_gain_ = 1.f;           // decoder.learnable_gain + 1
};

}  // namespace voc
