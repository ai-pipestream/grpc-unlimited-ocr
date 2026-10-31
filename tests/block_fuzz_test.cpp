// Fuzzed model output through the block parser: it never crashes, gives the
// same blocks whether the text arrives one token at a time or all at once,
// and its blocks cover every byte's step, in order, without gaps.

#include <random>

#include "check.h"
#include "output_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

// Fragments the model writes, plus the ways it gets them wrong.
const std::vector<std::string> kFragments = {
    "<BLOCK>",    "<FORK>",   "<MERGE>",  "<LEVEL1>",   "<LEVEL",   "< BLOCK >", "<|det|>",   "<image>",
    "text",       "title",    "table",    "cross_text", "append_table", "foo",   "image",    "equation",
    "[1,2,3,4]",  "[90, 67, 824, 99]",    "[-3,2000,5,6,7]", "[1,2",   "]",       "[1\xEF\xBC\x8C" "2,3,4]",
    "\n",         " ",        "word",     "\xE4\xB8\xAD\xE6\x96\x87", "<table><tr><td rowspan=\"2\">a</td></tr>",
    "</table>",   "<td>",     "<",        ">",          "&amp;",     "0",        "\xF0\x9F\x98\x80", "<PAGE>"};

std::string random_output(std::mt19937& rng) {
  std::string out;
  const int n = 1 + static_cast<int>(rng() % 60);
  for (int i = 0; i < n; ++i) {
    if (rng() % 8 == 0) {
      out.push_back(static_cast<char>(rng() % 256));  // raw bytes, invalid UTF-8 included
    } else {
      const std::string& f = kFragments[rng() % kFragments.size()];
      if (f != "<PAGE>") out += f;  // pages are split by the caller
    }
  }
  return out;
}

void check_coverage(const std::vector<v1::Block>& blocks, std::uint32_t steps) {
  require(!blocks.empty(), "a non-empty page has a block");
  require(blocks.front().first_step() == 0, "the first block starts at the first step");
  require(blocks.back().last_step() + 1 == steps, "the last block ends at the last step");
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    require(blocks[i].first_step() <= blocks[i].last_step(), "a block's steps are in order");
    require(blocks[i].ordinal() == i, "ordinals count up");
    if (i > 0) {
      // Neighbours meet on one step at most (a step can end one block and
      // start the next) and leave no step out.
      require(blocks[i].first_step() == blocks[i - 1].last_step() ||
                  blocks[i].first_step() == blocks[i - 1].last_step() + 1,
              "blocks cover the steps without gaps");
    }
  }
}

}  // namespace

int main() {
  return run_tests([] {
    std::mt19937 rng(20261030);
    for (int round = 0; round < 3000; ++round) {
      const std::string text = random_output(rng);
      if (text.empty()) continue;
      const auto whole = parse_whole(text);
      const auto pieces = tokenize(text, static_cast<unsigned>(round), 1 + round % 5);
      const auto stepped = parse_pieces(pieces);
      if (!same_blocks(stepped, whole)) {
        throw std::runtime_error("FAIL: incremental and whole parses differ for round " + std::to_string(round));
      }
      check_coverage(stepped, static_cast<std::uint32_t>(pieces.size()));
    }
  });
}
