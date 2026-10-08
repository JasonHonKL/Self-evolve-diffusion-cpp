#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spm {

// SentencePiece unigram (Viterbi) tokenizer for the UMT5 spiece format.
// Produces ids identical to HF AutoTokenizer on in-vocab input.
class SpmModel {
 public:
  bool load(const std::string& path);

  // HF-equivalent: normalize (collapse 2+ spaces), metaspace escape, Viterbi,
  // append eos. add_eos=false gives raw sentencepiece ids.
  std::vector<int> encode(std::string_view text, bool add_eos = true) const;

  int vocab_size() const { return (int)score_.size(); }
  int unk_id() const { return unk_id_; }
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
  int pad_id() const { return pad_id_; }
  bool byte_fallback() const { return byte_fallback_; }

 private:
  std::string blob_;  // whole .model file; piece keys are views into it
  std::unordered_map<std::string_view, int> piece_to_id_;
  std::vector<float> score_;
  std::vector<unsigned char> type_;  // 1 normal 2 unk 3 control 4 user 5 unused 6 byte
  size_t max_piece_len_ = 0;
  float min_score_ = 1e30f;
  int unk_id_ = 0, bos_id_ = 0, eos_id_ = 0, pad_id_ = 0;
  bool byte_fallback_ = false;
};

}  // namespace spm
