#include "service/document_session.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "output/document_fold.h"

namespace uocr::service {
namespace {

namespace docv1 = ::ai::pipestream::document::v1;

// A page whose model top choice the n-gram rule overrode this many steps
// in a row is reported as stuck in repetition.
constexpr std::uint32_t kRepetitionRun = 32;

v1::Intervention intervention(StepRule rule) {
  switch (rule) {
    case StepRule::kNone: return v1::INTERVENTION_NONE;
    case StepRule::kRepeatBan: return v1::INTERVENTION_REPEAT_BAN;
    case StepRule::kPageGuard: return v1::INTERVENTION_PAGE_GUARD;
  }
  return v1::INTERVENTION_UNSPECIFIED;
}

std::string finish_reason(WindowFinish finish) {
  switch (finish) {
    case WindowFinish::kEndOfSequence: return "stop";
    case WindowFinish::kTokenBudget: return "length";
    case WindowFinish::kContextLength: return "context_length";
    case WindowFinish::kStopped: return "stopped";
  }
  return {};
}

docv1::VlmStopReason stop_reason(WindowFinish finish) {
  switch (finish) {
    case WindowFinish::kEndOfSequence: return docv1::VLM_STOP_REASON_END_OF_SEQUENCE;
    case WindowFinish::kTokenBudget:
    case WindowFinish::kContextLength: return docv1::VLM_STOP_REASON_LENGTH;
    case WindowFinish::kStopped: return docv1::VLM_STOP_REASON_UNSPECIFIED;
  }
  return docv1::VLM_STOP_REASON_UNSPECIFIED;
}

v1::PageFailure failure(std::uint32_t page_no, v1::PageFailureCode code, std::string detail) {
  v1::PageFailure out;
  out.set_page_no(page_no);
  out.set_code(code);
  out.set_detail(std::move(detail));
  return out;
}

}  // namespace

struct DocumentSession::Page {
  explicit Page(const IntakePage& source) : source(source), parser(source.page_no) {
    tokens.set_page_no(source.page_no);
  }
  const IntakePage& source;
  output::BlockParser parser;
  v1::PageTokens tokens;
  std::size_t first_block = 0;  // where the page's blocks start in blocks_
  std::uint32_t ban_run = 0;
  bool repetition = false;
};

struct DocumentSession::Window {
  std::uint32_t index = 0;
  std::vector<IntakePage> pages;     // every page of the window, in order
  std::vector<std::size_t> decoded;  // the ones the engine reads (rasters that decoded)
  std::size_t cursor = 0;            // position in `decoded` of the open page
  std::size_t flushed = 0;           // pages[0, flushed) have had their events
  std::unique_ptr<Page> open;
  std::uint64_t prompt_tokens = 0;
};

DocumentSession::DocumentSession(Engine& engine, std::mutex& engine_mutex, const SessionOptions& options,
                                 PageIntake& intake, Emit emit, std::function<bool()> halted)
    : engine_(engine),
      engine_mutex_(engine_mutex),
      options_(options),
      intake_(intake),
      emit_(std::move(emit)),
      halted_(std::move(halted)) {}

DocumentSession::~DocumentSession() = default;

SessionEnd DocumentSession::run() {
  const std::uint32_t count = options_.page_count;
  const std::uint32_t window = std::max<std::uint32_t>(options_.page_window, 1);
  for (std::uint32_t index = 0, first = 1; first <= count; ++index, first += window) {
    const std::uint32_t last = std::min(count, first + window - 1);
    std::optional<std::vector<IntakePage>> pages = intake_.take(first, last);
    if (!pages.has_value()) return halted_() ? SessionEnd::kStopped : SessionEnd::kMissingPages;
    std::size_t bytes = 0;
    for (const IntakePage& page : *pages) bytes += page.raster.size();
    run_window(index, std::move(*pages));
    intake_.release(bytes);
    if (aborted_) return SessionEnd::kAborted;
    if (halted_()) return SessionEnd::kStopped;
  }
  complete();
  return SessionEnd::kComplete;
}

void DocumentSession::run_window(std::uint32_t index, std::vector<IntakePage> pages) {
  Window window;
  window.index = index;
  window.pages = std::move(pages);
  for (std::size_t i = 0; i < window.pages.size(); ++i) {
    const IntakePage& page = window.pages[i];
    if (!page.image_error.empty()) continue;
    window.decoded.push_back(i);
    geometry_.push_back({page.page_no, page.size.width_px, page.size.height_px});
  }
  if (window.decoded.empty()) {
    for (const IntakePage& page : window.pages) emit_failed_raster(page);
    return;
  }
  std::lock_guard engine_lock(engine_mutex_);
  decode_window(window);
}

void DocumentSession::decode_window(Window& window) {
  WindowRequest request;
  for (const std::size_t i : window.decoded) request.pages.emplace_back(window.pages[i].raster);
  request.tiled = options_.tiled;
  request.max_tokens_per_page = options_.max_tokens_per_page;
  open_next_page(window);
  const auto started = [&](std::uint64_t prompt_tokens) {
    window.prompt_tokens = prompt_tokens;
    prompt_tokens_ += prompt_tokens;
    v1::ParseDocumentResponse event;
    v1::WindowStarted* w = event.mutable_window_started();
    w->set_window_index(window.index);
    w->set_first_page_no(window.pages.front().page_no);
    w->set_last_page_no(window.pages.back().page_no);
    w->set_prompt_tokens(prompt_tokens);
    emit_(std::move(event));
  };
  const auto t0 = std::chrono::steady_clock::now();
  try {
    const WindowResult result = engine_.decode(request, started, [&](const EngineStep& s) { return on_step(window, s); });
    decode_time_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0);
    finish_window(window, result.finish, {});
  } catch (const std::exception& error) {
    decode_time_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0);
    finish_window(window, WindowFinish::kStopped, error.what());
  }
}

void DocumentSession::open_next_page(Window& window) {
  const std::size_t at = window.decoded[window.cursor];
  for (; window.flushed < at; ++window.flushed) emit_failed_raster(window.pages[window.flushed]);
  window.flushed = at + 1;
  window.open = std::make_unique<Page>(window.pages[at]);
  window.open->first_block = blocks_.size();
}

bool DocumentSession::on_step(Window& window, const EngineStep& step) {
  if (aborted_ || halted_()) return false;
  const std::uint32_t index = next_step_++;
  step_logprobs_.push_back(step.logprob);
  const bool marker = step.token == engine_.page_token();
  // A <PAGE> marker opens the next page; the page's own marker is its
  // first step. Markers past the window's last page stay on that page.
  if (marker && window.open->tokens.steps_size() > 0 && window.cursor + 1 < window.decoded.size()) {
    if (!close_page(window, *window.open, {}, false)) return false;
    ++window.cursor;
    open_next_page(window);
  }
  Page& page = *window.open;
  if (page.tokens.steps_size() == 0) page.tokens.set_first_step(index);
  v1::TokenStep* out = page.tokens.add_steps();
  out->set_token(step.token);
  out->set_piece(engine_.piece(step.token));
  out->set_logprob(step.logprob);
  out->set_model_token(step.model_token);
  out->set_model_logprob(step.model_logprob);
  out->set_intervention(intervention(step.rule));
  const std::string bytes = engine_.bytes(step.token);
  page.tokens.mutable_text()->append(bytes);
  if (!marker && step.token != engine_.eos_token()) handle_blocks(page.parser.add(index, bytes));
  page.ban_run = step.rule == StepRule::kRepeatBan ? page.ban_run + 1 : 0;
  page.repetition = page.repetition || page.ban_run >= kRepetitionRun;
  return !aborted_ && !halted_();
}

void DocumentSession::handle_blocks(std::vector<v1::Block> blocks) {
  for (v1::Block& block : blocks) {
    std::vector<v1::ContinuationLink> links = tracker_.add(block);
    if (options_.emit_blocks) {
      v1::ParseDocumentResponse event;
      *event.mutable_block() = block;
      emit_(std::move(event));
    }
    for (v1::ContinuationLink& link : links) {
      v1::ParseDocumentResponse event;
      *event.mutable_continuation() = link;
      emit_(std::move(event));
      if (options_.emit_document) links_.push_back(std::move(link));
    }
    if (options_.emit_document || options_.emit_page_documents) blocks_.push_back(std::move(block));
  }
}

bool DocumentSession::close_page(Window& window, Page& page, std::vector<v1::PageFailure> failures, bool capped) {
  const std::size_t first_block = page.first_block;
  handle_blocks(page.parser.finish());
  const std::uint32_t page_no = page.source.page_no;
  const auto steps = static_cast<std::uint32_t>(page.tokens.steps_size());
  if (!capped && options_.max_tokens_per_page > 0 && steps >= options_.max_tokens_per_page) {
    capped = true;
    failures.push_back(failure(page_no, v1::PAGE_FAILURE_CODE_TOKEN_CAP,
                               std::to_string(steps) + " steps reached the per-page cap of " +
                                   std::to_string(options_.max_tokens_per_page)));
  }
  if (page.repetition) {
    failures.push_back(failure(page_no, v1::PAGE_FAILURE_CODE_REPETITION,
                               "the no-repeat n-gram rule overrode the model " + std::to_string(kRepetitionRun) +
                                   " or more steps in a row"));
  }
  if (page.parser.unparseable()) {
    failures.push_back(failure(page_no, v1::PAGE_FAILURE_CODE_UNPARSEABLE_OUTPUT,
                               "the block grammar placed none of the page's " +
                                   std::to_string(page.parser.text().bytes().size()) + " bytes of output"));
  }
  for (v1::PageFailure& f : failures) {
    v1::ParseDocumentResponse event;
    *event.mutable_page_failure() = std::move(f);
    emit_(std::move(event));
  }
  (failures.empty() ? pages_ok_ : pages_failed_) += 1;
  v1::ParseDocumentResponse tokens;
  *tokens.mutable_tokens() = std::move(page.tokens);
  emit_(std::move(tokens));
  if (options_.emit_page_documents) {
    const output::PageGeometry geometry{page_no, page.source.size.width_px, page.source.size.height_px};
    const output::DocumentInput input{
        .name = "page-" + std::to_string(page_no),
        .version = options_.version,
        .pages = {geometry},
        .generation = {window.prompt_tokens, steps, capped ? docv1::VLM_STOP_REASON_LENGTH
                                                          : docv1::VLM_STOP_REASON_END_OF_SEQUENCE,
                       capped ? "length" : "stop"},
        .step_logprobs = step_logprobs_};
    const std::vector<v1::Block> page_blocks(blocks_.begin() + static_cast<std::ptrdiff_t>(first_block),
                                             blocks_.end());
    v1::ParseDocumentResponse event;
    event.mutable_page_document()->set_page_no(page_no);
    *event.mutable_page_document()->mutable_document() = output::page_document(input, page_blocks);
    emit_(std::move(event));
  }
  if (!options_.emit_document) blocks_.resize(first_block);
  if (!failures.empty() && options_.abort_on_error) aborted_ = true;
  return !aborted_;
}

void DocumentSession::emit_untouched_page(const IntakePage& page, v1::PageFailureCode code,
                                          const std::string& detail) {
  if (aborted_) return;
  v1::ParseDocumentResponse event;
  *event.mutable_page_failure() = failure(page.page_no, code, detail);
  emit_(std::move(event));
  v1::ParseDocumentResponse tokens;
  tokens.mutable_tokens()->set_page_no(page.page_no);
  tokens.mutable_tokens()->set_first_step(next_step_);
  emit_(std::move(tokens));
  ++pages_failed_;
  if (options_.abort_on_error) aborted_ = true;
}

void DocumentSession::emit_failed_raster(const IntakePage& page) {
  emit_untouched_page(page, v1::PAGE_FAILURE_CODE_IMAGE_DECODE, page.image_error);
}

void DocumentSession::finish_window(Window& window, WindowFinish finish, const std::string& engine_error) {
  last_finish_ = finish;
  if (aborted_ || halted_()) return;
  const bool capped = finish == WindowFinish::kTokenBudget || finish == WindowFinish::kContextLength;
  std::vector<v1::PageFailure> failures;
  const std::uint32_t page_no = window.open->source.page_no;
  if (!engine_error.empty()) {
    failures.push_back(failure(page_no, v1::PAGE_FAILURE_CODE_INFERENCE, "engine failed: " + engine_error));
  } else if (capped) {
    failures.push_back(failure(page_no, v1::PAGE_FAILURE_CODE_TOKEN_CAP, "the window's decode budget ran out"));
  }
  if (!close_page(window, *window.open, std::move(failures), capped)) return;
  window.open.reset();
  // Pages the decode never reached, and failed rasters among them.
  for (; window.flushed < window.pages.size(); ++window.flushed) {
    const IntakePage& page = window.pages[window.flushed];
    if (!page.image_error.empty()) {
      emit_failed_raster(page);
    } else if (!engine_error.empty()) {
      emit_untouched_page(page, v1::PAGE_FAILURE_CODE_INFERENCE, "engine failed before the page: " + engine_error);
    } else {
      emit_untouched_page(page, v1::PAGE_FAILURE_CODE_TOKEN_CAP, "the window's decode ended before the page began");
    }
  }
}

void DocumentSession::complete() {
  const output::GenerationFacts generation{prompt_tokens_, next_step_, stop_reason(last_finish_),
                                           finish_reason(last_finish_)};
  if (options_.emit_document) {
    const output::DocumentInput input{.name = "document",
                                      .version = options_.version,
                                      .pages = geometry_,
                                      .generation = generation,
                                      .step_logprobs = step_logprobs_};
    v1::ParseDocumentResponse event;
    *event.mutable_document() = output::fold_document(input, blocks_, links_);
    emit_(std::move(event));
  }
  v1::ParseDocumentResponse event;
  v1::ParseComplete* done = event.mutable_complete();
  done->set_pages_ok(pages_ok_);
  done->set_pages_failed(pages_failed_);
  done->set_prompt_tokens(prompt_tokens_);
  done->set_completion_tokens(next_step_);
  done->set_stop_reason(generation.stop_reason);
  done->set_decode_microseconds(static_cast<std::uint64_t>(decode_time_.count()));
  emit_(std::move(event));
}

}  // namespace uocr::service
