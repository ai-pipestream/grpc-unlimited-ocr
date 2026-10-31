#pragma once

// Shared by the output tests: run text through the incremental parser as
// tokens, and read the golden page's raw output.

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "fake_model.h"
#include "output/block_parser.h"

namespace uocr::testing {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

inline std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// One page's text through the parser, the pieces fed one step each.
inline std::vector<v1::Block> parse_pieces(const std::vector<std::string>& pieces, std::uint32_t page_no = 1,
                                           output::BlockParser* keep = nullptr) {
  output::BlockParser local(page_no);
  output::BlockParser& parser = keep != nullptr ? *keep : local;
  std::vector<v1::Block> out;
  std::uint32_t step = 0;
  for (const std::string& piece : pieces) {
    for (auto& b : parser.add(step++, piece)) out.push_back(std::move(b));
  }
  for (auto& b : parser.finish()) out.push_back(std::move(b));
  return out;
}

inline std::vector<v1::Block> parse_text(std::string_view text, unsigned seed = 1, int max_run = 4) {
  return parse_pieces(tokenize(text, seed, max_run));
}

// The same text as a single step.
inline std::vector<v1::Block> parse_whole(std::string_view text) { return parse_pieces({std::string(text)}); }

// Blocks equal apart from their step ranges (which depend on tokenisation).
inline bool same_blocks(std::vector<v1::Block> a, std::vector<v1::Block> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (auto* blk : {&a[i], &b[i]}) {
      blk->clear_first_step();
      blk->clear_last_step();
    }
    if (a[i].SerializeAsString() != b[i].SerializeAsString()) return false;
  }
  return true;
}

}  // namespace uocr::testing
