#include "output/block_parser.h"

#include <algorithm>
#include <array>
#include <utility>

#include "output/html_table.h"

namespace uocr::output {
namespace {

struct LabelKind {
  std::string_view label;
  v1::BlockKind kind;
};

// Upstream's labels; "picture" and "img" are the image aliases its
// renderer accepts.
constexpr std::array<LabelKind, 19> kLabels = {{
    {"text", v1::BLOCK_KIND_TEXT},
    {"title", v1::BLOCK_KIND_TITLE},
    {"header", v1::BLOCK_KIND_PAGE_HEADER},
    {"footer", v1::BLOCK_KIND_PAGE_FOOTER},
    {"page_number", v1::BLOCK_KIND_PAGE_NUMBER},
    {"image", v1::BLOCK_KIND_IMAGE},
    {"picture", v1::BLOCK_KIND_IMAGE},
    {"img", v1::BLOCK_KIND_IMAGE},
    {"figure", v1::BLOCK_KIND_FIGURE},
    {"chart", v1::BLOCK_KIND_CHART},
    {"seal", v1::BLOCK_KIND_SEAL},
    {"table", v1::BLOCK_KIND_TABLE},
    {"image_caption", v1::BLOCK_KIND_IMAGE_CAPTION},
    {"image_footnote", v1::BLOCK_KIND_IMAGE_FOOTNOTE},
    {"page_footnote", v1::BLOCK_KIND_PAGE_FOOTNOTE},
    {"equation", v1::BLOCK_KIND_EQUATION},
    {"ref_text", v1::BLOCK_KIND_REFERENCE},
    {"aside_text", v1::BLOCK_KIND_ASIDE},
    {"algorithm", v1::BLOCK_KIND_ALGORITHM},
}};

std::string lowered(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

// The label without its cross_ or append_ prefix, lowercased.
std::string base_label(std::string_view label) {
  std::string base = lowered(label);
  for (const std::string_view prefix : {"cross_", "append_"}) {
    if (base.starts_with(prefix)) return base.substr(prefix.size());
  }
  return base;
}

std::uint32_t clamp_coordinate(std::int32_t v) { return static_cast<std::uint32_t>(std::clamp(v, 0, 1000)); }

void fill_content(v1::Block& block, const BlockSpan& span, std::string_view content) {
  if (!span.recognised) {
    block.mutable_text()->set_text(std::string(content));  // verbatim
    return;
  }
  const std::string clean = clean_content(content);
  switch (block.kind()) {
    case v1::BLOCK_KIND_TABLE: {
      v1::TableContent* table = block.mutable_table();
      *table = parse_html_table(clean);
      table->set_source_html(std::string(trim(content)));
      return;
    }
    case v1::BLOCK_KIND_EQUATION:
      block.mutable_formula()->set_latex(clean);
      return;
    case v1::BLOCK_KIND_IMAGE:
    case v1::BLOCK_KIND_FIGURE:
    case v1::BLOCK_KIND_CHART:
    case v1::BLOCK_KIND_SEAL:
      block.mutable_picture()->set_text(clean);
      return;
    default:
      block.mutable_text()->set_text(clean);
  }
}

}  // namespace

v1::BlockKind block_kind(std::string_view label) {
  const std::string base = base_label(label);
  for (const auto& entry : kLabels) {
    if (entry.label == base) return entry.kind;
  }
  return v1::BLOCK_KIND_UNRECOGNISED;
}

v1::Continuation block_continuation(std::string_view label) {
  const std::string l = lowered(label);
  if (l.starts_with("cross_")) return v1::CONTINUATION_CONTINUES_ON_NEXT_PAGE;
  if (l.starts_with("append_")) return v1::CONTINUATION_CONTINUED_FROM_PREVIOUS;
  return v1::CONTINUATION_UNSPECIFIED;
}

v1::Block make_block(std::uint32_t page_no, std::uint32_t ordinal, const BlockSpan& span, const PageText& text) {
  v1::Block block;
  block.set_page_no(page_no);
  block.set_ordinal(ordinal);
  block.set_kind(span.recognised ? block_kind(span.label) : v1::BLOCK_KIND_UNRECOGNISED);
  block.set_continuation(span.recognised ? block_continuation(span.label) : v1::CONTINUATION_UNSPECIFIED);
  if (block.kind() == v1::BLOCK_KIND_TITLE && span.level.has_value() && *span.level >= 0) {
    block.set_heading_level(static_cast<std::uint32_t>(*span.level));
  }
  if (span.raw_box.size() >= 4) {
    v1::NormalizedBox* box = block.mutable_box();
    box->set_x1(clamp_coordinate(span.raw_box[0]));
    box->set_y1(clamp_coordinate(span.raw_box[1]));
    box->set_x2(clamp_coordinate(span.raw_box[2]));
    box->set_y2(clamp_coordinate(span.raw_box[3]));
  }
  block.set_merges_with_previous(span.merge_with_previous);
  block.set_label_text(span.label);
  for (const std::int32_t v : span.raw_box) block.add_raw_box(v);
  block.set_first_step(text.first_step(span.begin));
  block.set_last_step(text.last_step(span.end));
  fill_content(block, span, text.bytes().substr(span.content_begin, span.end - span.content_begin));
  return block;
}

std::vector<v1::Block> BlockParser::add(std::uint32_t step, std::string_view bytes) {
  if (bytes.empty()) return {};
  text_.append(step, bytes);
  // Spans before the last are final, so only the last one is parsed again,
  // from its own start; its <MERGE> flag depends on text before it and
  // stays as first found.
  const std::size_t base = spans_.empty() ? 0 : spans_.back().begin;
  std::vector<BlockSpan> fresh = split_blocks(text_.bytes().substr(base));
  if (!spans_.empty()) {
    const bool merge = spans_.back().merge_with_previous;
    spans_.pop_back();
    if (base > 0 && !fresh.empty()) fresh.front().merge_with_previous = merge;
  }
  for (BlockSpan& span : fresh) {
    span.begin += base;
    span.content_begin += base;
    span.end += base;
    spans_.push_back(std::move(span));
  }
  return take(spans_.size() - 1);
}

std::vector<v1::Block> BlockParser::finish() {
  const bool any_recognised = std::ranges::any_of(spans_, &BlockSpan::recognised);
  unparseable_ = !any_recognised && !clean_content(text_.bytes()).empty();
  return take(spans_.size());
}

std::vector<v1::Block> BlockParser::take(std::size_t count) {
  std::vector<v1::Block> out;
  for (; emitted_ < count; ++emitted_) {
    out.push_back(make_block(page_no_, static_cast<std::uint32_t>(emitted_), spans_[emitted_], text_));
  }
  return out;
}

}  // namespace uocr::output
