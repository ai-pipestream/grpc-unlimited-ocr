#pragma once

// Typed document checks for the real-model quickstart fixture. Validate the
// wire projection independently of DocumentBuilder, including all references.
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include <cmath>
#include <map>
#include <set>

#include "check.h"
#include "output/document_builder.h"

namespace uocr::testing {
namespace model_docs {
namespace v1 = ::ai::pipestream::unlimitedocr::v1;
namespace docv1 = ::ai::pipestream::document::v1;

inline std::string stable_bytes(const docv1::Document& doc) {
  std::string bytes;
  {
    google::protobuf::io::StringOutputStream sink(&bytes);
    google::protobuf::io::CodedOutputStream stream(&sink);
    stream.SetSerializationDeterministic(true);
    require(doc.SerializeToCodedStream(&stream), "serialize typed document");
  }
  return bytes;
}

inline void check_payload(const docv1::Document& doc, const std::vector<v1::Block>& blocks,
                          const std::vector<output::PageGeometry>& pages) {
  require(!blocks.empty() && doc.pages_size() == static_cast<int>(pages.size()), "document page set present");
  require(doc.body().self_ref() == "#/body" && doc.body().children_size() == static_cast<int>(blocks.size()),
          "document body covers every fixture block");
  std::map<unsigned, output::PageGeometry> geometry;
  for (const auto& page : pages) {
    require(geometry.emplace(page.page_no, page).second && doc.pages().contains(page.page_no), "unique page identity");
    const auto& got = doc.pages().at(page.page_no);
    require(got.page_no() == static_cast<int>(page.page_no) && got.size().width() == page.width_px &&
                got.size().height() == page.height_px,
            "document page dimensions match input");
  }
  int texts = 0, pictures = 0;
  std::map<unsigned, int> last_picture;
  std::map<int, std::vector<std::string>> captions;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    require(geometry.contains(block.page_no()) && block.has_box(), "block page and geometry present");
    const auto& page = geometry.at(block.page_no());
    const auto check_item = [&](const auto& item, const std::string& ref, docv1::DocItemLabel label) {
      require(item.self_ref() == ref && item.parent().ref() == "#/body" &&
                  doc.body().children(static_cast<int>(i)).ref() == ref,
              "body and item references agree");
      require(item.label() == label && item.prov_size() == 1, "item label and provenance count");
      const auto& prov = item.prov(0);
      const auto& box = prov.bbox();
      const auto coordinate = [](double got, double want) {
        // Independent pixel scaling may differ in double rounding only.
        return std::isfinite(got) && std::isfinite(want) && std::abs(got - want) <= 1e-9;
      };
      require(prov.page_no() == static_cast<int>(page.page_no) && box.coord_origin() == docv1::COORD_ORIGIN_TOPLEFT &&
                  coordinate(box.l(), block.box().x1() * static_cast<double>(page.width_px) / 1000.0) &&
                  coordinate(box.r(), block.box().x2() * static_cast<double>(page.width_px) / 1000.0) &&
                  coordinate(box.t(), block.box().y1() * static_cast<double>(page.height_px) / 1000.0) &&
                  coordinate(box.b(), block.box().y2() * static_cast<double>(page.height_px) / 1000.0),
              "item geometry matches typed block");
    };
    if (block.kind() == v1::BLOCK_KIND_IMAGE || block.kind() == v1::BLOCK_KIND_CHART) {
      require(pictures < doc.pictures_size(), "picture payload present");
      const auto& item = doc.pictures(pictures);
      check_item(item, "#/pictures/" + std::to_string(pictures),
                 block.kind() == v1::BLOCK_KIND_CHART ? docv1::DOC_ITEM_LABEL_CHART : docv1::DOC_ITEM_LABEL_PICTURE);
      require(item.children_size() == 0, "fixture picture has no invented child");
      last_picture[block.page_no()] = pictures++;
    } else {
      require(texts < doc.texts_size(), "text payload present");
      const auto& item = doc.texts(texts);
      const bool title = block.kind() == v1::BLOCK_KIND_TITLE;
      require((title && item.has_title()) || (!title && item.has_text()), "typed fixture text kind");
      const auto& base = title ? item.title().base() : item.text().base();
      const auto ref = "#/texts/" + std::to_string(texts++);
      check_item(base, ref,
                 title                                          ? docv1::DOC_ITEM_LABEL_TITLE
                 : block.kind() == v1::BLOCK_KIND_IMAGE_CAPTION ? docv1::DOC_ITEM_LABEL_CAPTION
                                                                : docv1::DOC_ITEM_LABEL_TEXT);
      require(base.text() == block.text().text() && base.orig() == block.text().text(), "complete document text");
      if (block.kind() == v1::BLOCK_KIND_IMAGE_CAPTION) {
        require(last_picture.contains(block.page_no()), "fixture caption follows picture on same page");
        captions[last_picture.at(block.page_no())].push_back(ref);
      }
    }
  }
  require(texts == doc.texts_size() && pictures == doc.pictures_size() && doc.tables_size() == 0,
          "no missing or extra fixture items");
  for (int i = 0; i < pictures; ++i) {
    const auto& refs = captions[i];
    require(doc.pictures(i).captions_size() == static_cast<int>(refs.size()), "caption reference count");
    for (std::size_t j = 0; j < refs.size(); ++j)
      require(doc.pictures(i).captions(static_cast<int>(j)).ref() == refs[j], "caption reference target");
  }
}

class StreamChecks {
 public:
  explicit StreamChecks(std::vector<output::PageGeometry> pages) : pages_(std::move(pages)) {}
  void observe(const v1::ParseDocumentResponse& event) {
    if (event.has_tokens()) {
      const auto page = event.tokens().page_no();
      require(valid(page) && token_pages_.insert(page).second, "unique token page identity");
    }
    if (event.has_page_document()) {
      const auto page = event.page_document().page_no();
      require(valid(page) && fragments_.emplace(page, event.page_document().document()).second,
              "unique fragment page identity");
    }
    if (event.has_document()) {
      require(!has_folded_, "exactly one folded document");
      folded_ = event.document();
      has_folded_ = true;
    }
  }
  std::vector<std::string> finish(const std::vector<v1::Block>& blocks) const {
    require(token_pages_.size() == pages_.size() && fragments_.size() == pages_.size() && has_folded_,
            "all requested token pages, fragments and folded document present");
    std::vector<std::string> snapshot;
    for (const auto& page : pages_) {
      std::vector<v1::Block> selected;
      for (const auto& block : blocks)
        if (block.page_no() == page.page_no) selected.push_back(block);
      const auto& fragment = fragments_.at(page.page_no);
      check_payload(fragment, selected, {page});
      snapshot.push_back(stable_bytes(fragment));
    }
    check_payload(folded_, blocks, pages_);
    snapshot.push_back(stable_bytes(folded_));
    return snapshot;
  }

 private:
  bool valid(unsigned page) const {
    for (const auto& p : pages_)
      if (p.page_no == page) return true;
    return false;
  }
  std::vector<output::PageGeometry> pages_;
  std::set<unsigned> token_pages_;
  std::map<unsigned, docv1::Document> fragments_;
  docv1::Document folded_;
  bool has_folded_ = false;
};
}  // namespace model_docs
}  // namespace uocr::testing
