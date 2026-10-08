// .sdcpp v1 mmap loader (issue #15). Format (ALL LITTLE-ENDIAN):
//   b'SDCP' | u32 version=1 | u32 tensor_count
//   per tensor: u32 name_len | name bytes | u8 dtype (0=int8 rowwise, 1=f32 raw)
//             | u32 ndim | i64 dims[ndim] | u64 data_offset (into data blob)
//   data blob (starts right after the last entry):
//     dtype 0 -> i8 weights [numel] then f32 scales [rows],
//                rows = numel/dims[-1]; scale[j] = amax(row j)/127
//     dtype 1 -> f32 [numel] raw (small tensors, numel < 65536, or preserved)
#include "serde/sdcpp.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sd {
namespace {

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  [[noreturn]] void fail(const char* m) const {
    throw std::runtime_error(std::string("sdcpp: ") + m);
  }
  void need(size_t n) const { if ((size_t)(end - p) < n) fail("truncated file"); }
  // ponytail: memcpy LE decode — fine on LE hosts (x86/aarch64); byteswap if BE port matters
  uint32_t u32() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
  uint64_t u64() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
  uint8_t u8() { need(1); return *p++; }
};

struct Entry {
  std::string name;
  uint8_t dtype;
  std::vector<int64_t> dims;
  uint64_t off;
  int64_t numel, rows, blob;
};

}  // namespace

std::unordered_map<std::string, Tensor> load_sdcpp(const std::string& path,
                                                   std::shared_ptr<void>& owner) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("sdcpp: cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 16) {
    close(fd);
    throw std::runtime_error("sdcpp: truncated file " + path);
  }
  size_t size = (size_t)st.st_size;
  void* base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED) throw std::runtime_error("sdcpp: mmap failed " + path);
  owner = std::shared_ptr<void>(base, [size](void* b) { munmap(b, size); });

  const uint8_t* b = (const uint8_t*)base;
  Reader r{b, b + size};
  if (std::memcmp(r.p, "SDCP", 4) != 0) r.fail("bad magic");
  r.p += 4;
  if (r.u32() != 1) r.fail("unsupported version");
  uint32_t count = r.u32();

  std::vector<Entry> es;
  es.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    Entry e;
    uint32_t nl = r.u32();
    r.need(nl);
    e.name.assign((const char*)r.p, nl);
    r.p += nl;
    e.dtype = r.u8();
    if (e.dtype > 1) r.fail("bad dtype code");
    uint32_t nd = r.u32();
    if (nd > 8) r.fail("ndim too large");
    e.dims.resize(nd);
    int64_t numel = 1;
    for (auto& d : e.dims) {
      d = (int64_t)r.u64();
      if (d < 0) r.fail("negative dim");
      if (d && numel > (int64_t{1} << 40) / d) r.fail("tensor too large");
      numel *= d;
    }
    e.numel = numel;
    e.off = r.u64();
    if (e.dtype == 0) {
      if (nd == 0 || e.dims.back() < 1) r.fail("quantized tensor needs a last dim");
      e.rows = numel / e.dims.back();
      e.blob = numel + e.rows * 4;   // i8 weights, then f32 scales
    } else {
      e.rows = 0;
      e.blob = numel * 4;
    }
    es.push_back(std::move(e));
  }

  uint64_t data_size = (uint64_t)(r.end - r.p);   // blob = rest of file
  for (const Entry& e : es)
    if (e.off > data_size || (uint64_t)e.blob > data_size - e.off)
      r.fail("data out of bounds");

  std::unordered_map<std::string, Tensor> out;
  for (Entry& e : es) {
    const uint8_t* w = r.p + e.off;
    Tensor t;
    t.shape = e.dims;
    t.owning = false;
    if (e.dtype == 0) {
      t.dtype = DType::I8;
      t.data = (void*)w;
      t.nbytes = (size_t)e.numel;
      out.emplace(e.name, std::move(t));
      Tensor s;
      s.shape = {e.rows};
      s.dtype = DType::F32;
      s.data = (void*)(w + e.numel);
      s.nbytes = (size_t)e.rows * 4;
      s.owning = false;
      out.emplace(e.name + ".scale", std::move(s));
    } else {
      t.dtype = DType::F32;
      t.data = (void*)w;
      t.nbytes = (size_t)e.blob;
      out.emplace(e.name, std::move(t));
    }
  }
  return out;
}

}  // namespace sd
