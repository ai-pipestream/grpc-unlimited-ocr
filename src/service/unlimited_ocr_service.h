#pragma once

// The synchronous gRPC service. Each ParseDocument stream runs three
// threads: the RPC thread reads pages into the stream's intake (stopping
// when the stream's byte budget is full, so gRPC flow control holds the
// client back), a session thread decodes windows as they complete, and a
// writer thread drains events onto the wire. Decoded events are queued
// without a bound: they are small next to the rasters, and a client that
// stops reading only delays its own stream.

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>

#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.grpc.pb.h"
#include "service/budget.h"
#include "service/config.h"
#include "service/document_session.h"
#include "service/engine.h"
#include "service/page_intake.h"

namespace uocr::service {

class UnlimitedOcrServiceImpl final : public ::ai::pipestream::unlimitedocr::v1::UnlimitedOcrService::Service {
 public:
  using Request = ::ai::pipestream::unlimitedocr::v1::ParseDocumentRequest;
  using Response = ::ai::pipestream::unlimitedocr::v1::ParseDocumentResponse;

  UnlimitedOcrServiceImpl(const Config& config, Engine& engine);

  // Transport-independent ParseDocument: `read` yields the next client
  // message (false at half-close), `write` sends one event (false when the
  // client is gone), `cancelled` mirrors ServerContext::IsCancelled.
  grpc::Status ParseDocumentCore(const std::function<bool(Request*)>& read,
                                 const std::function<bool(const Response&)>& write,
                                 const std::function<bool()>& cancelled);

  grpc::Status ParseDocument(grpc::ServerContext* context,
                             grpc::ServerReaderWriter<Response, Request>* stream) override;

  grpc::Status GetServiceInfo(grpc::ServerContext* context,
                              const ::ai::pipestream::unlimitedocr::v1::GetServiceInfoRequest* request,
                              ::ai::pipestream::unlimitedocr::v1::GetServiceInfoResponse* response) override;

  // Server-wide budgets, exposed for tests.
  const Budget& buffered_bytes() const { return buffered_bytes_; }
  const Budget& document_slots() const { return document_slots_; }

  std::atomic<long> documents_ok{0};
  std::atomic<long> documents_failed{0};

 private:
  // Checks one page and stores it once it fits the budgets. nullopt: it
  // was stored; OK: the stream stopped meanwhile; otherwise the error that
  // ends the stream.
  std::optional<grpc::Status> admit_page(::ai::pipestream::unlimitedocr::v1::PageImage& page, PageIntake& intake,
                                         const SessionOptions& options, const std::function<bool()>& halted);

  const Config& config_;
  Engine& engine_;
  std::mutex engine_mutex_;
  Budget buffered_bytes_;
  Budget document_slots_;
};

}  // namespace uocr::service
