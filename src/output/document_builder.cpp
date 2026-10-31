#include "output/document_builder.h"

#include <algorithm>
#include <cctype>
#include <tuple>

#include "output/block_grammar.h"

namespace uocr::output {
namespace {

// Past this many grid positions a table keeps its cells and leaves the
// grid empty: every covered position holds a copy of its anchor cell.
constexpr std::size_t kMaxGridPositions = 50000;
// Upstream drops at most this many repeated header rows from a continued
// table fragment.
constexpr std::size_t kMaxHeaderRows = 5;

std::string ref(const char* list, int index) { return "#/" + std::string(list) + "/" + std::to_string(index); }

bool is_picture(v1::BlockKind kind) {
  return kind == v1::BLOCK_KIND_IMAGE || kind == v1::BLOCK_KIND_FIGURE || kind == v1::BLOCK_KIND_CHART ||
         kind == v1::BLOCK_KIND_SEAL;
}

std::string block_text(const v1::Block& block) {
  switch (block.content_case()) {
    case v1::Block::kText: return block.text().text();
    case v1::Block::kFormula: return block.formula().latex();
    case v1::Block::kPicture: return block.picture().text();
    default: return {};
  }
}

// A table row's identity for header de-duplication, as upstream keys it:
// the cell count and the row's text with whitespace removed, full-width
// ASCII folded and lowercased.
using RowKey = std::tuple<std::size_t, std::string>;

std::string fold_row_text(const std::string& text) {
  std::string out;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto c = static_cast<unsigned char>(text[i]);
    // U+FF01..U+FF5E is EF BC 81..EF BD 9E in UTF-8.
    if (c == 0xEF && i + 2 < text.size()) {
      const auto b1 = static_cast<unsigned char>(text[i + 1]);
      const auto b2 = static_cast<unsigned char>(text[i + 2]);
      const unsigned cp = 0xF000u | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
      if (cp >= 0xFF01 && cp <= 0xFF5E) {
        out.push_back(static_cast<char>(std::tolower(static_cast<int>(cp - 0xFEE0))));
        i += 2;
        continue;
      }
    }
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') continue;
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

std::vector<RowKey> row_keys(const v1::TableContent& table) {
  std::vector<RowKey> keys(table.row_count());
  for (const auto& cell : table.cells()) {
    if (cell.row() >= keys.size()) continue;
    auto& [count, text] = keys[cell.row()];
    ++count;
    text += fold_row_text(cell.text());
  }
  return keys;
}

docv1::TableCell table_cell(const v1::TableCellContent& cell, std::uint32_t row_offset) {
  docv1::TableCell out;
  const auto row = static_cast<std::int32_t>(cell.row() + row_offset);
  const auto col = static_cast<std::int32_t>(cell.column());
  out.set_row_span(static_cast<std::int32_t>(cell.row_span()));
  out.set_col_span(static_cast<std::int32_t>(cell.col_span()));
  out.set_start_row_offset_idx(row);
  out.set_end_row_offset_idx(row + static_cast<std::int32_t>(cell.row_span()));
  out.set_start_col_offset_idx(col);
  out.set_end_col_offset_idx(col + static_cast<std::int32_t>(cell.col_span()));
  out.set_text(cell.text());
  out.set_column_header(cell.header());
  return out;
}

void fill_grid(docv1::TableData* data) {
  const auto rows = static_cast<std::size_t>(data->num_rows());
  const auto cols = static_cast<std::size_t>(data->num_cols());
  if (rows == 0 || cols == 0 || rows * cols > kMaxGridPositions) return;
  std::vector<const docv1::TableCell*> at(rows * cols, nullptr);
  for (const auto& cell : data->table_cells()) {
    for (auto r = static_cast<std::size_t>(cell.start_row_offset_idx());
         r < std::min<std::size_t>(rows, static_cast<std::size_t>(cell.end_row_offset_idx())); ++r) {
      for (auto c = static_cast<std::size_t>(cell.start_col_offset_idx());
           c < std::min<std::size_t>(cols, static_cast<std::size_t>(cell.end_col_offset_idx())); ++c) {
        if (at[r * cols + c] == nullptr) at[r * cols + c] = &cell;
      }
    }
  }
  for (std::size_t r = 0; r < rows; ++r) {
    docv1::TableRow* row = data->add_grid();
    for (std::size_t c = 0; c < cols; ++c) {
      docv1::TableCell* out = row->add_cells();
      if (at[r * cols + c] != nullptr) {
        *out = *at[r * cols + c];
        continue;
      }
      // An empty position: a 1 x 1 placeholder so the grid stays rectangular.
      out->set_row_span(1);
      out->set_col_span(1);
      out->set_start_row_offset_idx(static_cast<std::int32_t>(r));
      out->set_end_row_offset_idx(static_cast<std::int32_t>(r + 1));
      out->set_start_col_offset_idx(static_cast<std::int32_t>(c));
      out->set_end_col_offset_idx(static_cast<std::int32_t>(c + 1));
    }
  }
}

void set_text_base(docv1::TextItemBase* base, docv1::DocItemLabel label, docv1::ContentLayer layer,
                   const std::vector<docv1::ProvenanceItem>& prov, const std::vector<docv1::SourceType>& sources,
                   const std::string& text) {
  base->set_label(label);
  base->set_content_layer(layer);
  for (const auto& p : prov) *base->add_prov() = p;
  for (const auto& s : sources) *base->add_source() = s;
  base->set_orig(text);
  base->set_text(text);
}

docv1::TextItemBase* text_base(docv1::BaseTextItem& item) {
  switch (item.item_case()) {
    case docv1::BaseTextItem::kTitle: return item.mutable_title()->mutable_base();
    case docv1::BaseTextItem::kSectionHeader: return item.mutable_section_header()->mutable_base();
    case docv1::BaseTextItem::kFormula: return item.mutable_formula()->mutable_base();
    case docv1::BaseTextItem::kText: return item.mutable_text()->mutable_base();
    default: return nullptr;
  }
}

void stamp_text(docv1::BaseTextItem& item, const std::string& self, const std::string& parent) {
  if (item.item_case() == docv1::BaseTextItem::kCode) {
    item.mutable_code()->set_self_ref(self);
    item.mutable_code()->mutable_parent()->set_ref(parent);
  } else if (docv1::TextItemBase* base = text_base(item)) {
    base->set_self_ref(self);
    base->mutable_parent()->set_ref(parent);
  }
}

}  // namespace

std::string joined_text(const std::vector<const v1::Block*>& group) {
  std::string out;
  for (const v1::Block* block : group) out += block_text(*block);
  return out;
}

DocumentBuilder::DocumentBuilder(const DocumentInput& input) : input_(input) {}

docv1::ProvenanceItem DocumentBuilder::provenance(const v1::Block& block) const {
  docv1::ProvenanceItem prov;
  prov.set_page_no(static_cast<std::int32_t>(block.page_no()));
  double w = 0;
  double h = 0;
  for (const auto& page : input_.pages) {
    if (page.page_no == block.page_no()) {
      w = page.width_px;
      h = page.height_px;
    }
  }
  docv1::BoundingBox* box = prov.mutable_bbox();
  if (block.has_box()) {
    box->set_l(block.box().x1() * w / 1000.0);
    box->set_t(block.box().y1() * h / 1000.0);
    box->set_r(block.box().x2() * w / 1000.0);
    box->set_b(block.box().y2() * h / 1000.0);
  } else {
    box->set_r(w);  // no box written: the whole page, never an invented one
    box->set_b(h);
  }
  box->set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
  return prov;
}

std::vector<docv1::SourceType> DocumentBuilder::sources(const std::vector<const v1::Block*>& group) const {
  docv1::CollectorSource collector;
  collector.set_collector(kCollectorName);
  collector.set_model(kModelName);
  collector.set_version(input_.version);
  double total = 0;
  std::uint64_t samples = 0;
  for (const v1::Block* block : group) {
    for (std::uint32_t s = block->first_step(); s <= block->last_step() && s < input_.step_logprobs.size(); ++s) {
      total += input_.step_logprobs[s];
      ++samples;
    }
  }
  if (samples > 0) {
    collector.set_raw_score(total / static_cast<double>(samples));
    collector.set_raw_score_kind("mean_logprob");
    collector.set_raw_score_samples(samples);
  }
  docv1::GenerationSource generation;
  generation.set_model(kModelName);
  generation.set_prompt_tokens(input_.generation.prompt_tokens);
  generation.set_completion_tokens(input_.generation.completion_tokens);
  generation.set_stop_reason(input_.generation.stop_reason);
  if (!input_.generation.finish_reason.empty()) generation.set_finish_reason(input_.generation.finish_reason);
  generation.set_temperature(0.0);
  std::vector<docv1::SourceType> out(2);
  *out[0].mutable_collector() = std::move(collector);
  *out[1].mutable_generation() = std::move(generation);
  return out;
}

void DocumentBuilder::add_group(const std::vector<const v1::Block*>& group) {
  if (group.empty()) return;
  const v1::Block& head = *group.front();
  if (is_picture(head.kind())) {
    add_picture_item(group);
  } else if (head.kind() == v1::BLOCK_KIND_TABLE) {
    add_table_item(group);
  } else {
    add_text_item(group, joined_text(group));
  }
}

void DocumentBuilder::add_text_item(const std::vector<const v1::Block*>& group, const std::string& text) {
  const v1::Block& head = *group.front();
  std::vector<docv1::ProvenanceItem> prov;
  for (const v1::Block* block : group) prov.push_back(provenance(*block));
  const auto src = sources(group);
  docv1::BaseTextItem* item = doc_.add_texts();
  const int index = doc_.texts_size() - 1;
  auto body = docv1::CONTENT_LAYER_BODY;
  switch (head.kind()) {
    case v1::BLOCK_KIND_TITLE: {
      const std::uint32_t level = std::max<std::uint32_t>(1, head.has_heading_level() ? head.heading_level() : 1);
      if (level == 1) {
        set_text_base(item->mutable_title()->mutable_base(), docv1::DOC_ITEM_LABEL_TITLE, body, prov, src, text);
      } else {
        set_text_base(item->mutable_section_header()->mutable_base(), docv1::DOC_ITEM_LABEL_SECTION_HEADER, body,
                      prov, src, text);
        item->mutable_section_header()->set_level(static_cast<std::int32_t>(level));
      }
      break;
    }
    case v1::BLOCK_KIND_ALGORITHM: {
      docv1::CodeItem* code = item->mutable_code();
      code->set_label(docv1::DOC_ITEM_LABEL_CODE);
      code->set_content_layer(body);
      for (const auto& p : prov) *code->add_prov() = p;
      for (const auto& s : src) *code->add_source() = s;
      code->set_orig(text);
      code->set_text(text);
      break;
    }
    case v1::BLOCK_KIND_EQUATION:
      set_text_base(item->mutable_formula()->mutable_base(), docv1::DOC_ITEM_LABEL_FORMULA, body, prov, src, text);
      break;
    default: {
      auto label = docv1::DOC_ITEM_LABEL_TEXT;
      switch (head.kind()) {
        case v1::BLOCK_KIND_PAGE_HEADER: label = docv1::DOC_ITEM_LABEL_PAGE_HEADER; break;
        case v1::BLOCK_KIND_PAGE_FOOTER:
        case v1::BLOCK_KIND_PAGE_NUMBER: label = docv1::DOC_ITEM_LABEL_PAGE_FOOTER; break;
        case v1::BLOCK_KIND_IMAGE_CAPTION: label = docv1::DOC_ITEM_LABEL_CAPTION; break;
        case v1::BLOCK_KIND_IMAGE_FOOTNOTE:
        case v1::BLOCK_KIND_PAGE_FOOTNOTE: label = docv1::DOC_ITEM_LABEL_FOOTNOTE; break;
        case v1::BLOCK_KIND_REFERENCE: label = docv1::DOC_ITEM_LABEL_REFERENCE; break;
        default: break;
      }
      const bool furniture = label == docv1::DOC_ITEM_LABEL_PAGE_HEADER || label == docv1::DOC_ITEM_LABEL_PAGE_FOOTER;
      docv1::TextItemBase* base = item->mutable_text()->mutable_base();
      set_text_base(base, label, furniture ? docv1::CONTENT_LAYER_FURNITURE : body, prov, src, text);
      if (head.kind() == v1::BLOCK_KIND_UNRECOGNISED) {
        base->set_label_raw(head.label_text().empty() ? "unrecognised" : head.label_text());
      }
    }
  }
  order_.push_back({List::kText, index});
  if (head.kind() == v1::BLOCK_KIND_IMAGE_CAPTION) link_caption(head, index);
}

void DocumentBuilder::link_caption(const v1::Block& caption, int text_index) {
  if (last_picture_ >= 0 && last_picture_page_ == caption.page_no()) {
    doc_.mutable_pictures(last_picture_)->add_captions()->set_ref(ref("texts", text_index));
    return;
  }
  // No picture yet on this page: the caption waits for the next one here.
  if (pending_page_ != caption.page_no()) pending_captions_.clear();
  pending_page_ = caption.page_no();
  pending_captions_.push_back(text_index);
}

void DocumentBuilder::add_picture_item(const std::vector<const v1::Block*>& group) {
  const v1::Block& head = *group.front();
  docv1::PictureItem* picture = doc_.add_pictures();
  const int index = doc_.pictures_size() - 1;
  picture->set_label(head.kind() == v1::BLOCK_KIND_CHART ? docv1::DOC_ITEM_LABEL_CHART : docv1::DOC_ITEM_LABEL_PICTURE);
  picture->set_content_layer(docv1::CONTENT_LAYER_BODY);
  for (const v1::Block* block : group) *picture->add_prov() = provenance(*block);
  for (const auto& s : sources(group)) *picture->add_source() = s;
  if (head.kind() == v1::BLOCK_KIND_SEAL) picture->set_label_raw(head.label_text());
  order_.push_back({List::kPicture, index});
  if (pending_page_ == head.page_no()) {
    for (const int caption : pending_captions_) picture->add_captions()->set_ref(ref("texts", caption));
  }
  pending_captions_.clear();
  last_picture_ = index;
  last_picture_page_ = head.page_no();
  // Text the model wrote inside the region becomes the picture's child.
  if (const std::string text = joined_text(group); !text.empty()) {
    docv1::BaseTextItem* item = doc_.add_texts();
    std::vector<docv1::ProvenanceItem> prov;
    for (const v1::Block* block : group) prov.push_back(provenance(*block));
    set_text_base(item->mutable_text()->mutable_base(), docv1::DOC_ITEM_LABEL_TEXT, docv1::CONTENT_LAYER_BODY, prov,
                  sources(group), text);
    const std::string self = ref("texts", doc_.texts_size() - 1);
    stamp_text(*item, self, ref("pictures", index));
    picture->add_children()->set_ref(self);
  }
}

void DocumentBuilder::add_table_item(const std::vector<const v1::Block*>& group) {
  docv1::TableItem* table = doc_.add_tables();
  const int index = doc_.tables_size() - 1;
  table->set_label(docv1::DOC_ITEM_LABEL_TABLE);
  table->set_content_layer(docv1::CONTENT_LAYER_BODY);
  for (const v1::Block* block : group) *table->add_prov() = provenance(*block);
  for (const auto& s : sources(group)) *table->add_source() = s;
  docv1::TableData* data = table->mutable_data();
  std::vector<RowKey> header;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  for (const v1::Block* block : group) {
    if (block->content_case() != v1::Block::kTable) continue;
    const v1::TableContent& part = block->table();
    const std::vector<RowKey> keys = row_keys(part);
    // A continued fragment drops leading rows that repeat the first
    // fragment's header, as upstream's merge does.
    std::uint32_t skip = 0;
    if (rows > 0) {
      while (skip < keys.size() && skip < header.size() && skip < kMaxHeaderRows && keys[skip] == header[skip]) ++skip;
    } else {
      header = keys;
    }
    for (const auto& cell : part.cells()) {
      if (cell.row() < skip) continue;
      v1::TableCellContent shifted = cell;
      shifted.set_row(cell.row() - skip);
      *data->add_table_cells() = table_cell(shifted, rows);
    }
    rows += part.row_count() - skip;
    cols = std::max(cols, part.column_count());
  }
  data->set_num_rows(static_cast<std::int32_t>(rows));
  data->set_num_cols(static_cast<std::int32_t>(cols));
  fill_grid(data);
  order_.push_back({List::kTable, index});
}

docv1::Document DocumentBuilder::finish() {
  doc_.set_schema_name("docling_document_v2");
  doc_.set_version("1.10.0");
  doc_.set_name(input_.name);
  docv1::GroupItem* body = doc_.mutable_body();
  body->set_self_ref("#/body");
  body->set_content_layer(docv1::CONTENT_LAYER_BODY);
  for (const Child& child : order_) {
    std::string self;
    if (child.list == List::kText) {
      self = ref("texts", child.index);
      stamp_text(*doc_.mutable_texts(child.index), self, "#/body");
    } else if (child.list == List::kPicture) {
      self = ref("pictures", child.index);
      doc_.mutable_pictures(child.index)->set_self_ref(self);
      doc_.mutable_pictures(child.index)->mutable_parent()->set_ref("#/body");
    } else {
      self = ref("tables", child.index);
      doc_.mutable_tables(child.index)->set_self_ref(self);
      doc_.mutable_tables(child.index)->mutable_parent()->set_ref("#/body");
    }
    body->add_children()->set_ref(self);
  }
  for (const auto& page : input_.pages) {
    docv1::PageItem& item = (*doc_.mutable_pages())[static_cast<std::int32_t>(page.page_no)];
    item.set_page_no(static_cast<std::int32_t>(page.page_no));
    item.mutable_size()->set_width(page.width_px);
    item.mutable_size()->set_height(page.height_px);
  }
  return std::move(doc_);
}

}  // namespace uocr::output
