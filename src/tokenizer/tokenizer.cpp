#include "tokenizer/tokenizer.h"

#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace uocr {
namespace {

// GPT-2's byte-to-unicode alphabet: printable bytes map to themselves, the
// rest to code points from 256 up, in byte order.
std::unordered_map<std::uint32_t, std::uint8_t> byte_level_alphabet() {
  std::unordered_map<std::uint32_t, std::uint8_t> out;
  std::uint32_t next = 256;
  for (std::uint32_t b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    out.emplace(printable ? b : next++, static_cast<std::uint8_t>(b));
  }
  return out;
}

// Next code point of UTF-8 `s` at `i`; advances `i`. Invalid bytes yield
// U+FFFD and advance one byte.
std::uint32_t next_code_point(const std::string& s, std::size_t& i) {
  const auto c = static_cast<unsigned char>(s[i]);
  int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
  if (len == 0 || i + static_cast<std::size_t>(len) > s.size()) {
    ++i;
    return 0xFFFD;
  }
  std::uint32_t cp = len == 1 ? c : c & (0xFF >> (len + 1));
  for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3F);
  i += static_cast<std::size_t>(len);
  return cp;
}

}  // namespace

Tokenizer Tokenizer::load(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(std::format("{}: cannot open", path.string()));
  }
  const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const auto json = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded() || !json.contains("model") || json["model"].value("type", "") != "BPE") {
    throw std::runtime_error(std::format("{}: not a BPE tokenizer", path.string()));
  }
  Tokenizer tok;
  tok.byte_of_char_ = byte_level_alphabet();
  auto place = [&](std::int64_t id, const std::string& content, bool special) {
    if (id < 0) return;
    if (static_cast<std::size_t>(id) >= tok.pieces_.size()) {
      tok.pieces_.resize(static_cast<std::size_t>(id) + 1);
      tok.special_.resize(static_cast<std::size_t>(id) + 1, false);
    }
    tok.pieces_[static_cast<std::size_t>(id)] = content;
    tok.special_[static_cast<std::size_t>(id)] = special;
  };
  for (const auto& [content, id] : json["model"]["vocab"].items()) {
    place(id.get<std::int64_t>(), content, false);
  }
  for (const auto& added : json.value("added_tokens", nlohmann::json::array())) {
    const auto id = added.value("id", std::int64_t{-1});
    const auto content = added.value("content", std::string{});
    place(id, content, true);
    tok.special_ids_.emplace(content, id);
  }
  return tok;
}

const std::string& Tokenizer::piece(std::int64_t id) const {
  static const std::string kEmpty;
  return id >= 0 && static_cast<std::size_t>(id) < pieces_.size() ? pieces_[static_cast<std::size_t>(id)] : kEmpty;
}

std::int64_t Tokenizer::special_id(const std::string& content) const {
  const auto it = special_ids_.find(content);
  return it == special_ids_.end() ? -1 : it->second;
}

std::string Tokenizer::decode(const std::vector<std::int64_t>& ids) const {
  std::string out;
  for (const std::int64_t id : ids) {
    const std::string& p = piece(id);
    if (id >= 0 && static_cast<std::size_t>(id) < special_.size() && special_[static_cast<std::size_t>(id)]) {
      out += p;
      continue;
    }
    for (std::size_t i = 0; i < p.size();) {
      const std::uint32_t cp = next_code_point(p, i);
      const auto it = byte_of_char_.find(cp);
      if (it != byte_of_char_.end()) out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

}  // namespace uocr
