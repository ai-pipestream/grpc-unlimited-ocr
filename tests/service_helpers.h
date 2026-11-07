#pragma once

// Drives the service's transport-independent ParseDocument with an
// in-memory client: a list of requests to read, and every event written.

#include <atomic>
#include <string>
#include <vector>

#include "fake_model.h"
#include "service/unlimited_ocr_service.h"

namespace uocr::testing {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

struct Outcome {
  grpc::Status status;
  std::vector<v1::ParseDocumentResponse> events;

  template <typename Pred>
  std::vector<const v1::ParseDocumentResponse*> where(Pred pred) const {
    std::vector<const v1::ParseDocumentResponse*> out;
    for (const auto& e : events) {
      if (pred(e)) out.push_back(&e);
    }
    return out;
  }
  std::vector<v1::PageTokens> tokens() const {
    std::vector<v1::PageTokens> out;
    for (const auto& e : events) {
      if (e.has_tokens()) out.push_back(e.tokens());
    }
    return out;
  }
  std::vector<v1::Block> blocks() const {
    std::vector<v1::Block> out;
    for (const auto& e : events) {
      if (e.has_block()) out.push_back(e.block());
    }
    return out;
  }
  std::vector<v1::PageFailure> failures() const {
    std::vector<v1::PageFailure> out;
    for (const auto& e : events) {
      if (e.has_page_failure()) out.push_back(e.page_failure());
    }
    return out;
  }
};

inline v1::ParseDocumentRequest options_request(std::uint32_t pages, bool everything = true) {
  v1::ParseDocumentRequest r;
  v1::ParseOptions* o = r.mutable_options();
  o->set_page_count(pages);
  o->set_emit_blocks(everything);
  o->set_emit_page_documents(everything);
  o->set_emit_document(everything);
  return r;
}

inline v1::ParseDocumentRequest page_request(std::uint32_t page_no, std::string raster = "PNG 1000 2000") {
  v1::ParseDocumentRequest r;
  r.mutable_page()->set_page_no(page_no);
  r.mutable_page()->set_png(std::move(raster));
  return r;
}

// Requests for `pages` pages in order, after `options`.
inline std::vector<v1::ParseDocumentRequest> document(v1::ParseDocumentRequest options) {
  std::vector<v1::ParseDocumentRequest> out{options};
  for (std::uint32_t p = 1; p <= options.options().page_count(); ++p) out.push_back(page_request(p));
  return out;
}

// `write_limit`: the client goes away after that many events.
inline Outcome run(service::UnlimitedOcrServiceImpl& service, std::vector<v1::ParseDocumentRequest> requests,
                   int write_limit = -1) {
  Outcome out;
  std::size_t next = 0;
  out.status = service.ParseDocumentCore(
      [&](v1::ParseDocumentRequest* r) {
        if (next >= requests.size()) return false;
        *r = requests[next++];
        return true;
      },
      [&](const v1::ParseDocumentResponse& e) {
        if (write_limit >= 0 && static_cast<int>(out.events.size()) >= write_limit) return false;
        out.events.push_back(e);
        return true;
      },
      [] { return false; });
  return out;
}

// One page of model output with a title and a text block.
inline std::string simple_page(int n) {
  return "<PAGE><BLOCK>title<LEVEL1>[10,10,900,60]<FORK>Page " + std::to_string(n) +
         "\n<BLOCK>text[10,80,900,200]<FORK>Body of page " + std::to_string(n) + ".\n";
}

}  // namespace uocr::testing
