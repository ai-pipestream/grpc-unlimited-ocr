#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace uocr::service {

// Only admitted RPCs create this watchdog. It times blocked transport
// operations, not total request duration. TryCancel's wire status is
// CANCELLED; it cannot promise to preserve a handler's application status.
class TransportWatch {
 public:
  using Clock = std::chrono::steady_clock;
  TransportWatch(std::chrono::milliseconds idle, std::function<void()> cancel)
      : idle_(idle), cancel_(std::move(cancel)), thread_([this](std::stop_token token) { run(token); }) {}
  void reading(bool active) { mark(read_, active); }
  void writing(bool active) { mark(write_, active); }
  void decoding(bool active) {
    std::lock_guard lock(mutex_);
    decoding_ = active;
    if (read_) read_ = Clock::now();
  }
  void stop() {
    std::lock_guard lock(mutex_);
    if (!stopping_) stopping_ = Clock::now();
  }

 private:
  void mark(std::optional<Clock::time_point>& operation, bool active) {
    std::lock_guard lock(mutex_);
    operation = active ? std::optional(Clock::now()) : std::nullopt;
  }
  bool overdue(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    if (decoding_ && read_ && !stopping_) read_ = now;
    for (const auto& operation : {read_, write_}) {
      if (!operation) continue;
      if (stopping_) {
        // Give already queued typed failures/partial results a chance to
        // drain. A blocked Read or Write must still terminate.
        if (now - std::max(*operation, *stopping_) >= std::chrono::milliseconds(200)) return true;
      } else if (now - *operation >= idle_) {
        return true;
      }
    }
    return false;
  }
  void run(std::stop_token token) {
    while (!token.stop_requested()) {
      if (overdue(Clock::now())) { cancel_(); return; }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  std::chrono::milliseconds idle_;
  std::function<void()> cancel_;
  std::mutex mutex_;
  std::optional<Clock::time_point> read_, write_, stopping_;
  bool decoding_ = false;
  std::jthread thread_;  // joined before the state above is destroyed
};

}  // namespace uocr::service
