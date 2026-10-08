// safetensors mmap loader contract (issue #4, implemented in src/serde/)
#pragma once
#include "core/tensor.h"
#include <string>
#include <unordered_map>

namespace sd {

// Mmaps a .safetensors file and returns named tensors VIEWS into the mapping
// (dtype preserved: BF16 stays BF16 — call sd::to_f32 to convert).
// `owner` keeps the mapping alive; keep it alive as long as views are used.
struct SafetensorsFile {
  std::unordered_map<std::string, Tensor> tensors;
  std::shared_ptr<void> mapping;   // mmap owner (opaque)
  std::string path;
};

SafetensorsFile load_safetensors(const std::string& path);

// Write a .safetensors file (test helper; bf16/f32/i8/f16/i32/u8 supported).
void write_safetensors(const std::string& path,
                       const std::unordered_map<std::string, Tensor>& tensors);

}  // namespace sd
