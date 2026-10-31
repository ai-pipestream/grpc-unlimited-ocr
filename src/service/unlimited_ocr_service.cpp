#include "service/unlimited_ocr_service.h"

#include <algorithm>
#include <optional>
#include <thread>

#include "service/channel.h"
#include "service/document_session.h"
#include "service/page_intake.h"

#ifndef UOCR_VERSION
#define UOCR_VERSION "0.0.0"
#endif

namespace uocr::service {
namespace {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

// How often the writer asks the transport whether the client is still there.
constexpr auto kPollInterval = std::chrono::milliseconds(50);

grpc::Status invalid(const std::string& message) { return {grpc::StatusCode::INVALID_ARGUMENT, message}; }

// The options a stream runs with, or the reason they are refused.
std::optional<grpc::Status> resolve_options(const v1::ParseOptions& in, const Config& config, SessionOptions* out) {
  if (in.page_count() == 0) return invalid("ParseOptions.page_count is required");
  if (in.page_count() > config.max_pages) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "page_count " + std::to_string(in.page_count()) + " is above the server's " +
                            std::to_string(config.max_pages) + " page cap");
  }
  if (!v1::ViewMode_IsValid(in.view_mode())) {
    return invalid("unknown view_mode value: " + std::to_string(static_cast<int>(in.view_mode())));
  }
  out->page_count = in.page_count();
  out->page_window = in.page_window() == 0 ? config.page_window : std::min<std::uint32_t>(in.page_window(), 64);
  out->max_tokens_per_page = in.max_tokens_per_page() == 0
                                 ? config.max_tokens_per_page
                                 : std::min(in.max_tokens_per_page(), config.max_tokens_per_page);
  out->tiled = in.view_mode() == v1::VIEW_MODE_TILED;
  out->emit_blocks = in.emit_blocks();
  out->emit_page_documents = in.emit_page_documents();
  out->emit_document = in.emit_document();
  out->abort_on_error = in.abort_on_error();
  out->version = UOCR_VERSION;
  return std::nullopt;
}

}  // namespace

UnlimitedOcrServiceImpl::UnlimitedOcrServiceImpl(const Config& config, Engine& engine)
    : config_(config),
      engine_(engine),
      buffered_bytes_(config.max_buffered_bytes),
      document_slots_(config.documents_in_flight) {}

grpc::Status UnlimitedOcrServiceImpl::ParseDocument(grpc::ServerContext* context,
                                                    grpc::ServerReaderWriter<Response, Request>* stream) {
  return ParseDocumentCore([&](Request* r) { return stream->Read(r); },
                           [&](const Response& e) { return stream->Write(e); },
                           [&] { return context->IsCancelled(); });
}

grpc::Status UnlimitedOcrServiceImpl::ParseDocumentCore(const std::function<bool(Request*)>& read,
                                                        const std::function<bool(const Response&)>& write,
                                                        const std::function<bool()>& cancelled) {
  Request request;
  if (!read(&request) || !request.has_options()) return invalid("the first message must be ParseOptions");
  SessionOptions options;
  if (auto refused = resolve_options(request.options(), config_, &options)) return *refused;

  PageIntake intake(options.page_count, config_.max_stream_buffered_bytes);
  Channel<Response> events;
  std::atomic<bool> halt{false};
  std::atomic<bool> client_gone{false};
  auto halted = [&] { return halt.load(); };
  auto stop = [&] {
    halt = true;
    intake.halt();
  };

  std::thread writer([&] {
    Response event;
    while (true) {
      if (cancelled && cancelled()) {
        client_gone = true;
        stop();
        return;
      }
      const auto popped = events.pop_for(&event, kPollInterval);
      if (popped == Channel<Response>::Popped::kClosed) return;
      if (popped == Channel<Response>::Popped::kValue && !write(event)) {
        client_gone = true;
        stop();
        return;
      }
    }
  });

  SessionEnd end = SessionEnd::kStopped;
  std::thread session_thread([&] {
    const Lease slot = document_slots_.acquire(1, halted);
    if (!slot) return;
    DocumentSession session(engine_, engine_mutex_, options, intake,
                            [&](Response event) { events.push(std::move(event)); }, halted);
    end = session.run();
    if (end != SessionEnd::kComplete) stop();
  });

  grpc::Status status = grpc::Status::OK;
  while (!halted() && read(&request)) {
    if (request.has_options()) {
      status = invalid("ParseOptions must not repeat on the stream");
      break;
    }
    if (!request.has_page()) continue;
    std::optional<grpc::Status> refused = admit_page(*request.mutable_page(), intake, options, halted);
    if (refused.has_value()) {
      status = *refused;
      break;
    }
  }
  intake.end_of_input();
  if (!status.ok()) stop();
  session_thread.join();
  events.close();
  writer.join();

  if (!status.ok()) {
    ++documents_failed;
    return status;
  }
  if (client_gone.load() || (cancelled && cancelled())) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "the client cancelled the stream");
  }
  switch (end) {
    case SessionEnd::kComplete:
      ++documents_ok;
      return grpc::Status::OK;
    case SessionEnd::kAborted:
      ++documents_failed;
      return grpc::Status(grpc::StatusCode::ABORTED, "abort_on_error is set and a page failed");
    case SessionEnd::kMissingPages:
      ++documents_failed;
      return invalid("the stream ended before all " + std::to_string(options.page_count) + " pages arrived");
    case SessionEnd::kStopped:
      break;
  }
  return grpc::Status(grpc::StatusCode::CANCELLED, "the stream stopped");
}

std::optional<grpc::Status> UnlimitedOcrServiceImpl::admit_page(v1::PageImage& page, PageIntake& intake,
                                                                const SessionOptions& options,
                                                                const std::function<bool()>& halted) {
  if (page.page_no() == 0 || page.page_no() > options.page_count) {
    return invalid("page_no " + std::to_string(page.page_no()) + " is outside 1.." +
                   std::to_string(options.page_count));
  }
  if (intake.has(page.page_no())) return invalid("page " + std::to_string(page.page_no()) + " was sent twice");
  if (page.png().size() > config_.max_page_bytes) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "page " + std::to_string(page.page_no()) +
                                                                  " is above the " +
                                                                  std::to_string(config_.max_page_bytes) +
                                                                  " byte cap");
  }
  IntakePage in;
  in.page_no = page.page_no();
  in.raster = std::move(*page.mutable_png());
  try {
    in.size = engine_.probe(in.raster);
  } catch (const ImageError& error) {
    in.image_error = error.what();
  }
  if (page.width_px() != 0 && page.height_px() != 0) in.size = {page.width_px(), page.height_px()};
  const PageIntake::Room room = intake.wait_for_room(in.page_no, in.raster.size(), halted);
  if (room == PageIntake::Room::kStuck) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "pages sent ahead of the decoding window filled the stream's " +
                            std::to_string(config_.max_stream_buffered_bytes) +
                            " byte budget; send pages in window order");
  }
  if (room == PageIntake::Room::kHalted) return grpc::Status::OK;
  in.process_bytes = buffered_bytes_.acquire(in.raster.size(), halted);
  if (!in.process_bytes) {
    intake.release(in.raster.size());
    return grpc::Status::OK;
  }
  intake.add(std::move(in));
  return std::nullopt;
}

grpc::Status UnlimitedOcrServiceImpl::GetServiceInfo(grpc::ServerContext* /*context*/,
                                                     const v1::GetServiceInfoRequest* /*request*/,
                                                     v1::GetServiceInfoResponse* response) {
  const EngineInfo info = engine_.info();
  response->set_version(UOCR_VERSION);
  response->set_backend(info.backend);
  response->set_device(info.device);
  response->set_model("Unlimited-OCR-2");
  response->set_manifest_sha256(info.manifest_sha256);
  response->set_weight_format(info.weight_format);
  response->set_page_window(config_.page_window);
  response->set_max_tokens_per_page(config_.max_tokens_per_page);
  response->set_max_pages(config_.max_pages);
  response->set_max_page_bytes(config_.max_page_bytes);
  response->set_documents_in_flight(config_.documents_in_flight);
  v1::UiInfo* ui = response->mutable_ui();
  ui->set_title("Unlimited OCR");
  ui->set_path("/ui/unlimited-ocr");
  ui->set_description("Whole-document OCR with Unlimited-OCR 2 on a native engine");
  return grpc::Status::OK;
}

}  // namespace uocr::service
