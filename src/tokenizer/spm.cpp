#include "spm.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace spm {
namespace {

bool ReadVarint(std::string_view b, size_t& i, uint64_t& v) {
  v = 0;
  for (int s = 0; s < 64 && i < b.size(); s += 7) {
    uint8_t x = (uint8_t)b[i++];
    v |= (uint64_t)(x & 0x7f) << s;
    if (!(x & 0x80)) return true;
  }
  return false;
}

struct Field {
  int num, wire;
  uint64_t vint = 0;
  std::string_view bytes;  // wire 2
  float f = 0;             // wire 5
};

bool NextField(std::string_view b, size_t& i, Field& f) {
  if (i >= b.size()) return false;
  uint64_t tag;
  if (!ReadVarint(b, i, tag)) return false;
  f.num = (int)(tag >> 3);
  f.wire = (int)(tag & 7);
  switch (f.wire) {
    case 0: return ReadVarint(b, i, f.vint);
    case 1:
      if (i + 8 > b.size()) return false;
      i += 8;
      return true;
    case 2: {
      uint64_t len;
      if (!ReadVarint(b, i, len) || i + len > b.size()) return false;
      f.bytes = b.substr(i, len);
      i += len;
      return true;
    }
    case 5:
      if (i + 4 > b.size()) return false;
      memcpy(&f.f, b.data() + i, 4);
      i += 4;
      return true;
    default: return false;
  }
}

size_t Utf8Len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xe) return 3;
  return 4;
}

// HF fast-tokenizer normalizer for this model: Replace(" {2,}", " ").
// ponytail: literal-space collapse only; spm spec itself is `identity`.
std::string Normalize(std::string_view t) {
  std::string r;
  r.reserve(t.size());
  for (size_t i = 0; i < t.size();)
    if (t[i] == ' ') {
      r += ' ';
      while (i < t.size() && t[i] == ' ') ++i;
    } else {
      r += t[i++];
    }
  return r;
}

// " hello world" -> "\xe2\x96\x81hello\xe2\x96\x81world"
std::string MetaspaceEscape(std::string_view t) {
  static const char kSpace[] = "\xe2\x96\x81";  // U+2581
  std::string r;
  r.reserve(t.size() + 3);
  r += kSpace;
  for (char c : t) {
    if (c == ' ') {
      r += kSpace;
    } else {
      r += c;
    }
  }
  return r;
}

}  // namespace

bool SpmModel::load(const std::string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  long n = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  blob_.resize(n);
  size_t got = fread(blob_.data(), 1, n, fp);
  fclose(fp);
  if ((long)got != n) return false;

  int byte_pieces = 0;
  std::string_view b = blob_;
  size_t i = 0;
  Field f;
  while (NextField(b, i, f)) {
    if (f.num == 1) {  // SentencePiece
      std::string_view piece;
      float score = 0;
      int type = 1;
      size_t j = 0;
      Field g;
      while (NextField(f.bytes, j, g)) {
        if (g.num == 1) piece = g.bytes;
        else if (g.num == 2) score = g.f;
        else if (g.num == 3) type = (int)g.vint;
      }
      int id = (int)score_.size();
      if (type == 5) {  // UNUSED: keep id slot, not matchable
        score_.push_back(score);
        type_.push_back(type);
        continue;
      }
      if (type == 6 && piece.size() == 6 && piece.compare(0, 3, "<0x") == 0)
        ++byte_pieces;
      piece_to_id_[piece] = id;
      score_.push_back(score);
      type_.push_back((unsigned char)type);
      if (piece.size() > max_piece_len_) max_piece_len_ = piece.size();
      if (score < min_score_) min_score_ = score;
    } else if (f.num == 2) {  // TrainerSpec: 40 unk 41 bos 42 eos 43 pad
      size_t j = 0;
      Field g;
      while (NextField(f.bytes, j, g)) {
        if (g.num == 40) unk_id_ = (int)g.vint;
        else if (g.num == 41) bos_id_ = (int)g.vint;
        else if (g.num == 42) eos_id_ = (int)g.vint;
        else if (g.num == 43) pad_id_ = (int)g.vint;
      }
    }
    // field 3 (NormalizerSpec): identity, no precompiled charsmap; add_dummy_prefix
    // is the spm default (true) and HF Metaspace always prepends here. Ignored.
  }
  // ponytail: infer byte_fallback from vocab (256 <0xXX> pieces), skip the
  // trainer_spec flag field number.
  byte_fallback_ = byte_pieces == 256;
  return !score_.empty();
}

std::vector<int> SpmModel::encode(std::string_view text, bool add_eos) const {
  std::vector<int> out;
  const std::string s = MetaspaceEscape(Normalize(text));
  const size_t n = s.size();
  const float unk_score = min_score_ - 10.0f;

  std::vector<float> best(n + 1, -1e30f);
  std::vector<int> back_id(n + 1, -1), back_pos(n + 1, -1);
  best[0] = 0;

  for (size_t pos = 0; pos < n; pos += Utf8Len((unsigned char)s[pos])) {
    if (best[pos] == -1e30f) continue;
    const size_t rest = n - pos;
    const size_t clen = Utf8Len((unsigned char)s[pos]);
    bool has_single = false;
    // pieces starting at pos (all vocab pieces are whole-char aligned)
    for (size_t L = 1; L <= max_piece_len_ && L <= rest; ++L) {
      auto it = piece_to_id_.find(std::string_view(s.data() + pos, L));
      if (it == piece_to_id_.end()) continue;
      const int id = it->second;
      if (L == clen) has_single = true;
      // user-defined symbols get the spm bonus: 0.1 * (chars - 1)
      size_t nch = 0;
      if (type_[id] == 4)
        for (size_t k = 0; k < L;) { ++nch; k += Utf8Len((unsigned char)s[pos + k]); }
      const float sc = type_[id] == 4 ? 0.1f * (float)(nch - 1) : score_[id];
      const float cand = best[pos] + sc;
      if (cand > best[pos + L]) {
        best[pos + L] = cand;
        back_id[pos + L] = id;
        back_pos[pos + L] = (int)pos;
      }
    }
    if (!has_single) {
      bool all_bytes = byte_fallback_;
      if (all_bytes)
        for (size_t k = 0; k < clen; ++k) {
          char pbuf[8];
          unsigned char v = (unsigned char)s[pos + k];
          snprintf(pbuf, sizeof pbuf, "<0x%02X>", v);
          if (!piece_to_id_.count(std::string_view(pbuf, 6))) {
            all_bytes = false;
            break;
          }
        }
      if (all_bytes) {  // decompose the unknown char into <0xXX> pieces
        for (size_t k = 0; k < clen; ++k) {
          char pbuf[8];
          unsigned char v = (unsigned char)s[pos + k];
          snprintf(pbuf, sizeof pbuf, "<0x%02X>", v);
          int id = piece_to_id_.find(std::string_view(pbuf, 6))->second;
          const float cand = best[pos + k] + score_[id];
          if (cand > best[pos + k + 1]) {
            best[pos + k + 1] = cand;
            back_id[pos + k + 1] = id;
            back_pos[pos + k + 1] = (int)(pos + k);
          }
        }
      } else {  // unknown char node
        const float cand = best[pos] + unk_score;
        if (cand > best[pos + clen]) {
          best[pos + clen] = cand;
          back_id[pos + clen] = unk_id_;
          back_pos[pos + clen] = (int)pos;
        }
      }
    }
  }

  for (int p = (int)n; p > 0; p = back_pos[p])
    if (back_id[p] >= 0) out.push_back(back_id[p]);
  std::reverse(out.begin(), out.end());
  if (add_eos) out.push_back(eos_id_);
  return out;
}

}  // namespace spm
