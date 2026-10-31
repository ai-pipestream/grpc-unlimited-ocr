#pragma once

// The model's HTML tables as typed cells: <tr> rows of <td>/<th> cells with
// colspan and rowspan, placed on a grid the way a browser places them.
// Malformed span values are repaired as upstream does (the first number in
// the value, or 1). Only the cells come out; the HTML itself is kept
// verbatim beside them by the caller.

#include <string_view>

#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.h"

namespace uocr::output {

::ai::pipestream::unlimitedocr::v1::TableContent parse_html_table(std::string_view html);

}  // namespace uocr::output
