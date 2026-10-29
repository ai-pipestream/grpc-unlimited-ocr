#pragma once

// Decoding side of the model's byte-level BPE tokenizer, plus the special
// token ids the engine needs. Encoding is not required at run time: the
// prompt is special tokens plus one fixed instruction whose ids are pinned
// in prompt_builder.h and checked against this decoder by a test.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace uocr {

class Tokenizer {
 public:
  // Reads `tokenizer.json`. Throws std::runtime_error when it is not a
  // byte-level BPE tokenizer.
  static Tokenizer load(const std::filesystem::path& path);

  // UTF-8 text of `ids`; special tokens are kept as their literal content.
  std::string decode(const std::vector<std::int64_t>& ids) const;

  // The literal content of one id ("<PAGE>", "Ġpage", ...), empty when unknown.
  const std::string& piece(std::int64_t id) const;

  // Id of a special (added) token by content; -1 when absent.
  std::int64_t special_id(const std::string& content) const;

  std::size_t size() const { return pieces_.size(); }

 private:
  std::vector<std::string> pieces_;
  std::vector<bool> special_;
  std::unordered_map<std::string, std::int64_t> special_ids_;
  std::unordered_map<std::uint32_t, std::uint8_t> byte_of_char_;  // byte-level alphabet, inverted
};

}  // namespace uocr
