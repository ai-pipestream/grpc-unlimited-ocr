// Losslessness (design section 8): for every run, PageTokens cover each
// decode step exactly once and in order, the EOS steps included; their
// pieces and text decode back to the raw output byte for byte; every
// block's steps lie inside its page; and the blocks cover every text step.
// Checked on scripted documents and on fuzzed model output.

#include <random>
#include <set>

#include "check.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

void check_run(const Outcome& out, const std::vector<WindowScript>& windows, std::uint32_t pages) {
  require(out.status.ok(), "the run completes: " + out.status.error_message());
  const auto tokens = out.tokens();
  require(tokens.size() == pages, "one PageTokens per page");
  std::string expected;
  for (const auto& w : windows) expected += w.output + std::string(w.eos ? kEosText : "");
  std::string text;
  std::string pieces;
  std::uint32_t next = 0;
  for (std::uint32_t i = 0; i < pages; ++i) {
    const auto& t = tokens[i];
    require(t.page_no() == i + 1, "pages close in order");
    if (t.steps_size() > 0) require(t.first_step() == next, "steps are contiguous across pages");
    next += static_cast<std::uint32_t>(t.steps_size());
    text += t.text();
    for (const auto& s : t.steps()) pieces += s.piece();
  }
  require(text == expected, "PageTokens text is the raw output byte for byte");
  require(pieces == expected, "the pieces decode back to the raw output");
  const auto& complete = out.events.back();
  require(complete.has_complete() && complete.complete().completion_tokens() == next, "every step is counted");

  // Blocks: inside their page, and covering every step but the markers.
  std::map<std::uint32_t, std::set<std::uint32_t>> covered;
  for (const auto& b : out.blocks()) {
    const auto& page = tokens[b.page_no() - 1];
    const std::uint32_t first = page.first_step();
    const auto last = first + static_cast<std::uint32_t>(page.steps_size());
    require(b.first_step() >= first && b.last_step() < last, "a block's steps lie inside its page");
    for (std::uint32_t s = b.first_step(); s <= b.last_step(); ++s) covered[b.page_no()].insert(s);
  }
  for (const auto& page : tokens) {
    for (int i = 0; i < page.steps_size(); ++i) {
      const auto token = page.steps(i).token();
      if (token == kPageToken || token == kEosToken) continue;
      require(covered[page.page_no()].contains(page.first_step() + static_cast<std::uint32_t>(i)),
              "every text step is covered by a block");
    }
  }
}

Outcome run_script(const std::vector<WindowScript>& windows, std::uint32_t pages, std::uint32_t window) {
  service::Config config;
  config.page_window = window;
  FakeEngine engine;
  engine.windows = windows;
  service::UnlimitedOcrServiceImpl service(config, engine);
  return run(service, document(options_request(pages)));
}

void scripted() {
  std::vector<WindowScript> windows(2);
  windows[0].output = simple_page(1) + "<PAGE>stray before<BLOCK>cross_table[1,2,3,4]<FORK><table><tr><td>a</td></tr>";
  windows[1].output = "<PAGE><BLOCK>append_table[1,2,3,4]<FORK><tr><td>b</td></tr></table>" + simple_page(4);
  check_run(run_script(windows, 4, 2), windows, 4);
}

std::string fuzzed_page(std::mt19937& rng) {
  static const std::vector<std::string> parts = {
      "<BLOCK>", "<FORK>", "<MERGE>", "<LEVEL2>", "text", "title", "table", "append_text", "[1,2,3,4]",
      "[5,6", "]", "\n", "words ", "\xE6\x96\x87", "<table><tr><td>x</td></tr></table>", "<|det|>", "foo"};
  std::string out = "<PAGE>";
  const int n = static_cast<int>(rng() % 40);
  for (int i = 0; i < n; ++i) out += parts[rng() % parts.size()];
  return out;
}

void fuzzed() {
  std::mt19937 rng(77);
  for (int round = 0; round < 200; ++round) {
    const std::uint32_t pages = 1 + static_cast<std::uint32_t>(rng() % 5);
    const std::uint32_t window = 1 + static_cast<std::uint32_t>(rng() % 3);
    std::vector<WindowScript> windows;
    for (std::uint32_t first = 1; first <= pages; first += window) {
      WindowScript w;
      for (std::uint32_t p = first; p <= std::min(pages, first + window - 1); ++p) w.output += fuzzed_page(rng);
      windows.push_back(w);
    }
    check_run(run_script(windows, pages, window), windows, pages);
  }
}

}  // namespace

int main() {
  return run_tests([] {
    scripted();
    fuzzed();
  });
}
