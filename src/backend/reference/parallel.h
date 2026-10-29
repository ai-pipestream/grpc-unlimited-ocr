#pragma once

// Splits an index range across threads. Each index is computed by exactly
// one thread with the same arithmetic as a serial loop, so results do not
// depend on the thread count. A parallel_for inside another runs serially
// on the calling thread instead of multiplying threads.

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace uocr::reference {

// Threads the reference backend uses; 1 forces serial execution.
unsigned thread_count();
void set_thread_count(unsigned threads);

namespace detail {
inline thread_local bool in_parallel_region = false;
}

template <typename Fn>
void parallel_for(std::int64_t count, Fn&& fn) {
  const auto threads = std::min<std::int64_t>(thread_count(), count);
  if (threads <= 1 || detail::in_parallel_region) {
    for (std::int64_t i = 0; i < count; ++i) {
      fn(i);
    }
    return;
  }
  std::vector<std::jthread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (std::int64_t t = 0; t < threads; ++t) {
    pool.emplace_back([&fn, t, threads, count] {
      detail::in_parallel_region = true;
      for (std::int64_t i = t; i < count; i += threads) {
        fn(i);
      }
    });
  }
}

}  // namespace uocr::reference
