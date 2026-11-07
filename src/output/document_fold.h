#pragma once

// The two Documents the service sends: a page's fragment, one item per
// block and nothing joined, and the folded document, where blocks the
// continuation links join become one item (a table across pages, a
// paragraph that continues, merged columns) with provenance on every page
// it spans.

#include <vector>

#include "output/document_builder.h"

namespace uocr::output {

docv1::Document page_document(const DocumentInput& input, const std::vector<v1::Block>& blocks);

docv1::Document fold_document(const DocumentInput& input, const std::vector<v1::Block>& blocks,
                              const std::vector<v1::ContinuationLink>& links);

}  // namespace uocr::output
