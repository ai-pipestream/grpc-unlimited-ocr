#include "output/page_text.h"

namespace uocr::output {

void PageText::append(std::uint32_t step, std::string_view bytes) {
  bytes_.append(bytes);
  step_of_byte_.insert(step_of_byte_.end(), bytes.size(), step);
}

}  // namespace uocr::output
