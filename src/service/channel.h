#pragma once

// A closed FIFO between threads: producers push, one consumer pops until
// the channel is closed and drained. Adapted from grpc-vlm-convert.

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace uocr::service {

template <typename T>
class Channel {
 public:
  explicit Channel(std::size_t max_bytes = 16ULL * 1024 * 1024) : max_bytes_(max_bytes) {}

  bool push(T value) {
    const std::size_t bytes = value.SpaceUsedLong() + sizeof(T);
    if (bytes > max_bytes_) throw std::length_error("response exceeds output byte budget");
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [&] { return closed_ || bytes <= max_bytes_ - held_; });
      if (closed_) return false;
      held_ += bytes;
      queue_.push_back(std::move(value));
    }
    ready_.notify_one();
    return true;
  }

  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    ready_.notify_all();
  }

  enum class Popped { kValue, kClosed, kTimeout };

  // Waits up to `timeout` for a value; kClosed once closed and drained.
  Popped pop_for(T* value, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, timeout, [&] { return closed_ || !queue_.empty(); })) return Popped::kTimeout;
    if (queue_.empty()) return Popped::kClosed;
    held_ -= queue_.front().SpaceUsedLong() + sizeof(T);
    *value = std::move(queue_.front());
    queue_.pop_front();
    ready_.notify_all();
    return Popped::kValue;
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<T> queue_;
  bool closed_ = false;
  const std::size_t max_bytes_;
  std::size_t held_ = 0;
};

}  // namespace uocr::service
