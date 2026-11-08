#include "backend/metal/context_impl.h"
#include "backend/metal/ops.h"
#include "check.h"

using namespace uocr;

int main() {
  if (metal::device_count() == 0) return skip("no Metal device");
  return run_tests([] {
    const metal::Context ctx(0, 4096);
    require_throws([&] { metal::Array invalid(ctx, static_cast<metal::DType>(255), 1); },
                   "unknown dtype rejected before allocation");
    require(ctx.allocated_bytes() == 0, "invalid dtype did not allocate");
    metal::Array output(ctx, metal::DType::kF32, 128);
    output.zero();
    id<MTLCommandBuffer> pending;
    {
      metal::Array temporary(ctx, metal::DType::kF32, 128);
      const std::vector<float> values(128, 0.25f);
      temporary.upload(values.data(), 128);
      metal::copy(ctx, output.f32(), temporary.f32(), 128);
      pending = ctx.impl().command;
      // Inject a sticky error while encoded work still references temporary.
      ctx.impl().failure = std::make_exception_ptr(std::runtime_error("injected failure"));
    }
    require(ctx.allocated_bytes() == 1024, "pending storage retained despite failure");
    require_throws([&] { ctx.synchronize(); }, "sticky failure reported after draining");
    require(pending.status == MTLCommandBufferStatusCompleted, "pending work completed before retirement");
    require(ctx.impl().command == nil && ctx.impl().retired.empty(), "pending storage drained");
    require(ctx.allocated_bytes() == 512, "retired allocation released after completion");
    for (int i = 0; i < 128; ++i) require(output.f32()[i] == 0.25f, "retired buffer remained valid");
    require_throws([&] { ctx.synchronize(); }, "failure remains sticky");
  });
}
