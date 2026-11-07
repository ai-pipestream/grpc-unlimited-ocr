#pragma once

// The pages one stream has sent and the decoder has not used yet. The read
// loop adds pages as they arrive, in any order; the document session takes
// a window's pages once they are all in. A stream holds at most its byte
// budget of rasters. While a window decodes, later pages wait for room.
// If the pending window cannot fit, admission fails instead of waiting
// for that incomplete window to release its own bytes.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "service/budget.h"
#include "service/engine.h"

namespace uocr::service {

struct IntakePage {
  std::uint32_t page_no = 0;
  std::string raster;
  PageSize size;
  std::string image_error;  // set when the engine could not read the raster
  Lease process_bytes;      // this page's share of the server-wide budget
};

class PageIntake {
 public:
  PageIntake(std::uint32_t page_count, std::size_t stream_budget_bytes)
      : page_count_(page_count), budget_(stream_budget_bytes) {}

  enum class Room { kAdmitted, kHalted, kStuck };

  // Blocks until a raster of `bytes` for `page_no` fits the stream's
  // budget. kStuck: the page cannot fit, or the budget is full while the
  // session waits for an incomplete window, so nothing can free room.
  Room wait_for_room(std::uint32_t page_no, std::size_t bytes, const std::function<bool()>& halted);

  // Stores a page admitted by wait_for_room.
  void add(IntakePage page);
  bool has(std::uint32_t page_no) const;

  // The client half-closed: no more pages will come.
  void end_of_input();

  // Wakes every waiter; called when the stream stops.
  void halt();

  // Blocks until pages first..last are all in, then hands them over in
  // order. Empty when input ended without them or the stream stopped.
  std::optional<std::vector<IntakePage>> take(std::uint32_t first, std::uint32_t last);

  // Gives back the stream budget a taken window held, once it is decoded.
  void release(std::size_t bytes);

  std::uint32_t page_count() const { return page_count_; }

 private:
  bool window_ready() const;

  const std::uint32_t page_count_;
  const std::size_t budget_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::map<std::uint32_t, IntakePage> pages_;
  std::set<std::uint32_t> received_;
  std::size_t held_ = 0;
  std::uint32_t waiting_first_ = 0;  // the window the session waits for; 0 when it is not waiting
  std::uint32_t waiting_last_ = 0;
  bool input_ended_ = false;
  bool halted_ = false;
};

}  // namespace uocr::service
