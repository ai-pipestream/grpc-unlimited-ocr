#pragma once

// Blocks as fleet Document items (design 5.2): one item per group of
// blocks, every item with provenance on each page it spans, the collector
// source, the generation that produced it, and the model's own
// log-probability over the steps behind it.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ai/pipestream/document/v1/document.pb.h"
#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.h"

namespace uocr::output {

namespace docv1 = ::ai::pipestream::document::v1;
namespace v1 = ::ai::pipestream::unlimitedocr::v1;

inline constexpr const char* kCollectorName = "unlimited-ocr";
inline constexpr const char* kModelName = "Unlimited-OCR-2";

struct PageGeometry {
  std::uint32_t page_no = 0;
  std::uint32_t width_px = 0;
  std::uint32_t height_px = 0;
};

struct GenerationFacts {
  std::uint64_t prompt_tokens = 0;
  std::uint64_t completion_tokens = 0;
  docv1::VlmStopReason stop_reason = docv1::VLM_STOP_REASON_UNSPECIFIED;
  std::string finish_reason;  // verbatim companion of stop_reason
};

// Everything one Document is built from.
struct DocumentInput {
  std::string name;
  std::string version;  // this server's
  std::vector<PageGeometry> pages;
  GenerationFacts generation;
  // The log-probability of every decode step, by document-wide step
  // index; items carry the mean over their steps.
  std::span<const float> step_logprobs;
};

// Adds items to a Document in reading order and finishes its skeleton.
class DocumentBuilder {
 public:
  explicit DocumentBuilder(const DocumentInput& input);

  // One item from a group of blocks that the model wrote as one (a single
  // block, a cross-page chain, or merged columns); `group` is in reading
  // order and its first block decides the item's kind.
  void add_group(const std::vector<const v1::Block*>& group);

  // Sets refs, the body's children and the page entries; returns the
  // Document.
  docv1::Document finish();

 private:
  enum class List { kText, kPicture, kTable };
  struct Child {
    List list;
    int index;
  };

  docv1::ProvenanceItem provenance(const v1::Block& block) const;
  std::vector<docv1::SourceType> sources(const std::vector<const v1::Block*>& group) const;
  void add_text_item(const std::vector<const v1::Block*>& group, const std::string& text);
  void add_picture_item(const std::vector<const v1::Block*>& group);
  void add_table_item(const std::vector<const v1::Block*>& group);
  void link_caption(const v1::Block& caption, int text_index);

  const DocumentInput& input_;
  docv1::Document doc_;
  std::vector<Child> order_;
  int last_picture_ = -1;               // most recent picture on the current page
  std::uint32_t last_picture_page_ = 0;
  std::vector<int> pending_captions_;   // captions on this page before any picture
  std::uint32_t pending_page_ = 0;
};

// The text a group of blocks contributes, joined as upstream joins
// continued text: no separator.
std::string joined_text(const std::vector<const v1::Block*>& group);

}  // namespace uocr::output
