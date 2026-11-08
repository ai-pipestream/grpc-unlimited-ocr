#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <map>
#include <exception>
#include <string>
#include <initializer_list>
#include "backend/metal/context.h"

namespace uocr::metal {
struct Binding { id<MTLBuffer> buffer; std::size_t offset; };
struct Context::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  id<MTLCommandBuffer> command;
  id<MTLBuffer> dummy;
  std::map<std::uintptr_t, id<MTLBuffer>> buffers;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
  std::vector<id<MTLBuffer>> retired;
  std::size_t retired_bytes = 0;
  std::size_t cap = 0, allocated = 0, peak = 0;
  std::uint64_t kernels = 0, gemms = 0;
  unsigned pending = 0;
  std::exception_ptr failure;
  Binding binding(const void* ptr) const;
  id<MTLCommandBuffer> commands();
  void finish();
};
// Fixed scalar ABI shared with kernels.metal; integers are 64-bit.
struct Params { std::int64_t i[20]{}; float f[4]{}; };
void dispatch(const Context& ctx, const char* name, std::size_t threads,
              std::initializer_list<const void*> buffers, const Params& params);
}  // namespace uocr::metal
