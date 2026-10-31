#include "service/page_intake.h"

#include <algorithm>
#include <chrono>

namespace uocr::service {
namespace {

// How often a blocked reader looks at its stream's halt flag, which
// nothing signals.
constexpr auto kPollInterval = std::chrono::milliseconds(50);

}  // namespace

PageIntake::Room PageIntake::wait_for_room(std::uint32_t page_no, std::size_t bytes,
                                           const std::function<bool()>& halted) {
  std::unique_lock lock(mutex_);
  while (true) {
    if (halted_ || (halted && halted())) return Room::kHalted;
    const bool needed = waiting_first_ != 0 && page_no >= waiting_first_ && page_no <= waiting_last_;
    if (needed || held_ == 0 || held_ + bytes <= budget_) {
      held_ += bytes;
      return Room::kAdmitted;
    }
    if (waiting_first_ != 0 && !window_ready()) return Room::kStuck;
    changed_.wait_for(lock, kPollInterval);
  }
}

void PageIntake::add(IntakePage page) {
  {
    std::lock_guard lock(mutex_);
    const std::uint32_t page_no = page.page_no;
    pages_.emplace(page_no, std::move(page));
  }
  changed_.notify_all();
}

bool PageIntake::has(std::uint32_t page_no) const {
  std::lock_guard lock(mutex_);
  return pages_.contains(page_no);
}

void PageIntake::end_of_input() {
  {
    std::lock_guard lock(mutex_);
    input_ended_ = true;
  }
  changed_.notify_all();
}

void PageIntake::halt() {
  {
    std::lock_guard lock(mutex_);
    halted_ = true;
  }
  changed_.notify_all();
}

bool PageIntake::window_ready() const {
  for (std::uint32_t p = waiting_first_; p <= waiting_last_; ++p) {
    if (!pages_.contains(p)) return false;
  }
  return true;
}

std::optional<std::vector<IntakePage>> PageIntake::take(std::uint32_t first, std::uint32_t last) {
  std::unique_lock lock(mutex_);
  waiting_first_ = first;
  waiting_last_ = last;
  changed_.notify_all();  // a reader holding a page of this window may now let it in
  changed_.wait(lock, [&] { return halted_ || input_ended_ || window_ready(); });
  const bool ready = !halted_ && window_ready();
  waiting_first_ = 0;
  waiting_last_ = 0;
  if (!ready) return std::nullopt;
  std::vector<IntakePage> out;
  for (std::uint32_t p = first; p <= last; ++p) {
    auto node = pages_.extract(p);
    out.push_back(std::move(node.mapped()));
  }
  return out;
}

void PageIntake::release(std::size_t bytes) {
  {
    std::lock_guard lock(mutex_);
    held_ -= std::min(held_, bytes);
  }
  changed_.notify_all();
}

}  // namespace uocr::service
