#include "runtime/logit_policy.h"

#include <algorithm>
#include <limits>
#include <span>

namespace uocr {
namespace {

constexpr float kBanned = -std::numeric_limits<float>::infinity();

void keep_only(std::vector<float>& logits, std::int64_t token) {
  const float kept = logits[static_cast<std::size_t>(token)];
  std::ranges::fill(logits, kBanned);
  logits[static_cast<std::size_t>(token)] = kept;
}

}  // namespace

void LogitPolicy::ban_repeats(std::vector<float>& logits, const std::vector<std::int64_t>& generated) const {
  const auto n = static_cast<std::int64_t>(generated.size());
  const std::int64_t size = config_.ngram_size;
  if (size <= 0 || n < size) return;
  const std::span<const std::int64_t> prefix(generated.data() + n - (size - 1), static_cast<std::size_t>(size - 1));
  const std::int64_t start = std::max<std::int64_t>(0, n - config_.ngram_window);
  for (std::int64_t i = start; i <= n - size; ++i) {
    if (std::ranges::equal(std::span(generated.data() + i, static_cast<std::size_t>(size - 1)), prefix)) {
      const std::int64_t banned = generated[static_cast<std::size_t>(i + size - 1)];
      if (!config_.ngram_whitelist.contains(banned)) logits[static_cast<std::size_t>(banned)] = kBanned;
    }
  }
}

void LogitPolicy::guard_pages(std::vector<float>& logits, const std::vector<std::int64_t>& generated) {
  if (config_.expected_pages <= 0) return;
  const auto n = static_cast<std::int64_t>(generated.size());
  if (n > scanned_) {
    std::int64_t last_marker = -1;
    for (std::int64_t i = scanned_; i < n; ++i) {
      if (generated[static_cast<std::size_t>(i)] == config_.page_token) {
        ++markers_;
        last_marker = i;
      }
    }
    page_len_ = last_marker >= 0 ? n - 1 - last_marker : page_len_ + (n - scanned_);
    scanned_ = n;
  }
  if (markers_ > config_.expected_pages) {
    keep_only(logits, config_.eos_token);
  } else if (config_.max_page_tokens > 0 && page_len_ >= config_.max_page_tokens) {
    keep_only(logits, markers_ < config_.expected_pages ? config_.page_token : config_.eos_token);
  } else if (markers_ < config_.expected_pages) {
    logits[static_cast<std::size_t>(config_.eos_token)] = kBanned;
  }
}

std::int64_t LogitPolicy::choose(std::vector<float>& logits, const std::vector<std::int64_t>& generated) {
  ban_repeats(logits, generated);
  guard_pages(logits, generated);
  return std::distance(logits.begin(), std::ranges::max_element(logits));
}

}  // namespace uocr
