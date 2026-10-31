#pragma once

// A closed FIFO between threads: producers push, one consumer pops until
// the channel is closed and drained. Adapted from grpc-vlm-convert.

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace uocr::service {

template <typename T>
class Channel {
 public:
  void push(T value) {
    {
      std::lock_guard lock(mutex_);
      queue_.push_back(std::move(value));
    }
    ready_.notify_one();
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
    *value = std::move(queue_.front());
    queue_.pop_front();
    return Popped::kValue;
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<T> queue_;
  bool closed_ = false;
};

}  // namespace uocr::service
