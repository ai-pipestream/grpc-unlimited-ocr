#pragma once

// The decoded bytes of one page's output, and for every byte the decode
// step it came from, so a block found in the text points back at the exact
// steps the model produced it in.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace uocr::output {

class PageText {
 public:
  // Appends the bytes one decode step produced. A step that produced no
  // bytes leaves no trace here (the step itself is kept in PageTokens).
  void append(std::uint32_t step, std::string_view bytes);

  std::string_view bytes() const { return bytes_; }
  bool empty() const { return bytes_.empty(); }

  // The step that produced byte `index`; `index` must be in range.
  std::uint32_t step_at(std::size_t index) const { return step_of_byte_[index]; }

  // First and last step behind the bytes [begin, end); `end` > `begin`.
  std::uint32_t first_step(std::size_t begin) const { return step_at(begin); }
  std::uint32_t last_step(std::size_t end) const { return step_at(end - 1); }

 private:
  std::string bytes_;
  std::vector<std::uint32_t> step_of_byte_;
};

}  // namespace uocr::output
