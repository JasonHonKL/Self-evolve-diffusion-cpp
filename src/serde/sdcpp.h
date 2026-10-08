// .sdcpp v1 loader contract (issue #15). Format written by
// tools/quantize/pack_sdcpp.py — canonical layout comment lives there and in
// the .cpp: all little-endian, rowwise-int8 weights + f32 per-row scales.
#pragma once
#include "core/tensor.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace sd {

// Quantized tensors appear as an I8 view under `name` (original shape) plus an
// F32 per-row scale view under `name.scale` (rows = numel/last_dim; dequant
// w[r][c] = q[r][c] * scale[r]). Preserved tensors are a single F32 view.
// `owner` receives the mmap handle; keep it alive as long as views are used.
// Throws std::runtime_error on bad magic/version/truncation/out-of-bounds.
std::unordered_map<std::string, Tensor> load_sdcpp(const std::string& path,
                                                   std::shared_ptr<void>& owner);

}  // namespace sd
