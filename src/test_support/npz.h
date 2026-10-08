// Minimal NPZ reader for golden tests. STORED (method 0) entries only —
// golden files MUST be written with np.savez (np.savez_compressed writes DEFLATE).
// All dtypes convert to F32, except '|u1'/'|i1' load as I8 (u1 keeps raw bit pattern).
// '<i2'/'<i4'/'<i8' -> F32 (values > 2^24 lose precision). Throws std::runtime_error
// on anything unsupported (compressed entries, fortran_order, other dtypes).
#pragma once
#include <string>
#include <unordered_map>
#include "core/tensor.h"

namespace sd {

std::unordered_map<std::string, Tensor> load_npz(const std::string& path);

}  // namespace sd
