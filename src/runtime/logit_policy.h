#pragma once

// What happens to the logits between the model and the chosen token:
// an optional no-repeat n-gram ban over a sliding window of the output,
// and the page-count guard that makes the output carry exactly one <PAGE>
// per input page (EOS banned until then, forced after; a page that runs
// past its token cap is closed). Then greedy argmax.

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace uocr {

struct LogitPolicyConfig {
  std::int64_t ngram_size = 0;     // 0 disables the ban
  std::int64_t ngram_window = 100;
  std::unordered_set<std::int64_t> ngram_whitelist;
  std::int64_t expected_pages = 0;  // 0 disables the guard
  std::int64_t max_page_tokens = 20000;
  std::int64_t page_token = 128827;
  std::int64_t eos_token = 1;
};

class LogitPolicy {
 public:
  explicit LogitPolicy(LogitPolicyConfig config) : config_(std::move(config)) {}

  // Applies the ban and the guard to `logits` given the tokens generated
  // so far, then returns the greedy choice (lowest id on a tie).
  std::int64_t choose(std::vector<float>& logits, const std::vector<std::int64_t>& generated);

 private:
  void ban_repeats(std::vector<float>& logits, const std::vector<std::int64_t>& generated) const;
  void guard_pages(std::vector<float>& logits, const std::vector<std::int64_t>& generated);

  LogitPolicyConfig config_;
  std::int64_t markers_ = 0;
  std::int64_t scanned_ = 0;
  std::int64_t page_len_ = 0;
};

}  // namespace uocr
