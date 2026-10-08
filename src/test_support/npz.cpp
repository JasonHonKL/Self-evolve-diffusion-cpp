#include "test_support/npz.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sd {
namespace {

uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct NpyArray {
  std::vector<int64_t> shape;
  std::string descr;
  bool fortran = false;
  const uint8_t* data = nullptr;
  size_t data_nbytes = 0;
};

NpyArray parse_npy(const uint8_t* p, size_t n) {
  if (n < 10 || memcmp(p, "\x93NUMPY", 6) != 0) throw std::runtime_error("npz: bad npy magic");
  bool v1 = p[6] == 1;  // v1.x: u16 header len at 8; v2.x: u32 at 10
  size_t hlen = v1 ? rd16(p + 8) : rd32(p + 10);
  size_t hsize = (v1 ? 10 : 14) + hlen;
  if (hsize > n) throw std::runtime_error("npz: npy header truncated");
  std::string hdr((const char*)p + (v1 ? 10 : 14), hlen);

  NpyArray a;
  {  // descr: value of 'descr' key
    size_t k = hdr.find("'descr'");
    size_t q1 = k == std::string::npos ? std::string::npos : hdr.find('\'', k + 7);
    size_t q2 = q1 == std::string::npos ? std::string::npos : hdr.find('\'', q1 + 1);
    if (q2 == std::string::npos) throw std::runtime_error("npz: no descr");
    a.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
  }
  {  // fortran_order
    size_t k = hdr.find("'fortran_order'");
    if (k == std::string::npos) throw std::runtime_error("npz: no fortran_order");
    a.fortran = hdr.find("True", k) < hdr.find("False", k);
  }
  {  // shape: tuple of ints after 'shape'
    size_t k = hdr.find("'shape'");
    size_t p1 = k == std::string::npos ? std::string::npos : hdr.find('(', k);
    size_t p2 = p1 == std::string::npos ? std::string::npos : hdr.find(')', p1);
    if (p2 == std::string::npos) throw std::runtime_error("npz: no shape");
    int64_t cur = -1;
    for (size_t i = p1 + 1; i <= p2; i++) {
      char c = i < p2 ? hdr[i] : ',';
      if (c >= '0' && c <= '9') cur = (cur < 0 ? 0 : cur) * 10 + (c - '0');
      else if (cur >= 0) { a.shape.push_back(cur); cur = -1; }
    }
  }
  size_t item = 0;
  if (a.descr == "<f4" || a.descr == "<i4" || a.descr == "<i8") item = a.descr == "<i8" ? 8 : 4;
  else if (a.descr == "<i2") item = 2;
  else if (a.descr == "|u1" || a.descr == "|i1") item = 1;
  else throw std::runtime_error("npz: unsupported dtype " + a.descr);
  int64_t numel = 1; for (auto d : a.shape) numel *= d;
  if (a.fortran && a.shape.size() > 1 && numel > 1)
    throw std::runtime_error("npz: fortran_order unsupported (pass C-contiguous arrays)");  // ponytail
  if ((size_t)numel * item > n - hsize) throw std::runtime_error("npz: npy data truncated");
  a.data = p + hsize;
  a.data_nbytes = (size_t)numel * item;
  return a;
}

Tensor to_tensor(const NpyArray& a) {
  int64_t numel = 1; for (auto d : a.shape) numel *= d;
  if (a.descr == "<f4") {
    Tensor t(a.shape, DType::F32);
    memcpy(t.data, a.data, a.data_nbytes);
    return t;
  }
  if (a.descr == "|u1" || a.descr == "|i1") {  // raw bytes, u1 keeps its bit pattern
    Tensor t(a.shape, DType::I8);
    memcpy(t.data, a.data, a.data_nbytes);
    return t;
  }
  Tensor t(a.shape, DType::F32);  // <i2/<i4/<i8 -> F32
  float* d = t.ptr<float>();
  if (a.descr == "<i2") {
    const int16_t* s = (const int16_t*)a.data;
    for (int64_t i = 0; i < numel; i++) d[i] = (float)s[i];
  } else if (a.descr == "<i4") {
    const int32_t* s = (const int32_t*)a.data;
    for (int64_t i = 0; i < numel; i++) d[i] = (float)s[i];
  } else {
    const int64_t* s = (const int64_t*)a.data;
    for (int64_t i = 0; i < numel; i++) d[i] = (float)s[i];
  }
  return t;
}

}  // namespace

std::unordered_map<std::string, Tensor> load_npz(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("npz: cannot open " + path);
  std::vector<uint8_t> buf;
  { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); throw std::runtime_error("npz: ftell failed"); }
    buf.resize((size_t)n);
    if (n > 0 && fread(buf.data(), 1, (size_t)n, f) != (size_t)n) { fclose(f); throw std::runtime_error("npz: short read"); } }
  fclose(f);
  size_t n = buf.size();
  if (n < 22) throw std::runtime_error("npz: not a zip");

  size_t lo = n - std::min<size_t>(n, 22 + 65535), eocd = SIZE_MAX;  // EOCD in last 64K+22
  for (size_t i = n - 21; i-- > lo;)
    if (rd32(&buf[i]) == 0x06054b50) { eocd = i; break; }
  if (eocd == SIZE_MAX) throw std::runtime_error("npz: no EOCD");
  uint16_t count = rd16(&buf[eocd + 10]);
  size_t cd = rd32(&buf[eocd + 16]);

  std::unordered_map<std::string, Tensor> out;
  for (uint16_t e = 0; e < count; e++) {
    if (cd + 46 > n || rd32(&buf[cd]) != 0x02014b50) throw std::runtime_error("npz: bad central dir");
    uint16_t method = rd16(&buf[cd + 10]);
    uint32_t usize = rd32(&buf[cd + 24]);
    uint16_t fnlen = rd16(&buf[cd + 28]), extralen = rd16(&buf[cd + 30]), cmtlen = rd16(&buf[cd + 32]);
    uint32_t lho = rd32(&buf[cd + 42]);
    std::string name((const char*)&buf[cd + 46], fnlen);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
    if (method != 0)
      throw std::runtime_error("npz: entry " + name + " is compressed; use np.savez, not savez_compressed");
    if (lho + 30 > n || rd32(&buf[lho]) != 0x04034b50) throw std::runtime_error("npz: bad local header");
    size_t doff = lho + 30 + rd16(&buf[lho + 26]) + rd16(&buf[lho + 28]);
    if (doff + usize > n) throw std::runtime_error("npz: entry data truncated");
    out.emplace(std::move(name), to_tensor(parse_npy(&buf[doff], usize)));
    cd += 46 + (size_t)fnlen + extralen + cmtlen;
  }
  return out;
}

}  // namespace sd
