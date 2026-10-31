#pragma once

// One document through the engine, independent of the transport: windows
// of pages are taken from the intake as they complete, decoded, and every
// decode step is turned into events as it is chosen. Steps are assigned to
// pages by the model's own <PAGE> markers; each page closes when the next
// one opens (or its window ends), and pages close strictly in order.
//
// Per page, in order: its Blocks and ContinuationLinks as they close, any
// PageFailure, its PageTokens (every step, always), then its PageDocument.
// After the last window: the folded Document when asked for, then
// ParseComplete.

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.h"
#include "output/block_parser.h"
#include "output/continuations.h"
#include "output/document_builder.h"
#include "service/engine.h"
#include "service/page_intake.h"

namespace uocr::service {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

struct SessionOptions {
  std::uint32_t page_count = 0;
  std::uint32_t page_window = 0;
  std::uint32_t max_tokens_per_page = 0;
  bool tiled = false;
  bool emit_blocks = false;
  bool emit_page_documents = false;
  bool emit_document = false;
  bool abort_on_error = false;
  std::string version;
};

enum class SessionEnd {
  kComplete,      // ParseComplete was sent
  kAborted,       // abort_on_error and a page failed
  kStopped,       // the stream halted (client gone or cancelled)
  kMissingPages,  // input ended without every page
};

class DocumentSession {
 public:
  using Emit = std::function<void(v1::ParseDocumentResponse)>;

  // `engine_mutex` serialises windows of every stream on the one engine.
  DocumentSession(Engine& engine, std::mutex& engine_mutex, const SessionOptions& options, PageIntake& intake,
                  Emit emit, std::function<bool()> halted);
  ~DocumentSession();

  SessionEnd run();

 private:
  struct Page;
  struct Window;

  void run_window(std::uint32_t index, std::vector<IntakePage> pages);
  void decode_window(Window& window);
  bool on_step(Window& window, const EngineStep& step);
  void open_next_page(Window& window);
  void handle_blocks(std::vector<v1::Block> blocks);
  bool close_page(Window& window, Page& page, std::vector<v1::PageFailure> failures, bool capped);
  void emit_failed_raster(const IntakePage& page);
  void emit_untouched_page(const IntakePage& page, v1::PageFailureCode code, const std::string& detail);
  void finish_window(Window& window, WindowFinish finish, const std::string& engine_error);
  void complete();

  Engine& engine_;
  std::mutex& engine_mutex_;
  const SessionOptions options_;
  PageIntake& intake_;
  Emit emit_;
  std::function<bool()> halted_;

  output::ContinuationTracker tracker_;
  std::vector<v1::Block> blocks_;  // every block, for the folded Document
  std::vector<v1::ContinuationLink> links_;
  std::vector<output::PageGeometry> geometry_;
  std::vector<float> step_logprobs_;  // by document-wide step
  std::uint32_t next_step_ = 0;
  std::uint64_t prompt_tokens_ = 0;
  std::uint32_t pages_ok_ = 0;
  std::uint32_t pages_failed_ = 0;
  WindowFinish last_finish_ = WindowFinish::kEndOfSequence;
  std::chrono::microseconds decode_time_{0};
  bool aborted_ = false;
};

}  // namespace uocr::service
