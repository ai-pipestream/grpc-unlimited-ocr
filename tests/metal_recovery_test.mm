#include <cmath>
#include <random>
#include "backend/metal/context_impl.h"
#include "backend/reference/parallel.h"
#include "check.h"
#include "model/metal/metal_decoder.h"
#include "model/decoder.h"
#include "runtime/generator.h"
#include "runtime/prompt_builder.h"
#include "tensor/dtype.h"

using namespace uocr;
namespace uocr::metal {
struct MetalDecoderTestAccess {
  static std::int64_t capacity(const MetalDecoder& d) { return d.capacity_; }
  static bool partial_growth(const MetalDecoder& d, std::int64_t old_capacity) {
    const auto width = d.dims_.heads * d.dims_.head_dim;
    return d.capacity_ == old_capacity && d.cache_.front().k.count() > old_capacity * width &&
           d.cache_.front().v.count() == old_capacity * width;
  }
  static bool cleared(const MetalDecoder& d) {
    return d.cache_.empty() && !d.block_min_.count() && !d.block_max_.count() && !d.prompt_blocks_ &&
           !d.capacity_ && !d.capacity_blocks_ && !d.length_ && !d.one_coord_.tok && !d.one_coord_.page;
  }
};
}
namespace {
DecoderDims tiny() {
  DecoderDims d;
  d.vocab = 128830; d.hidden = 32; d.layers = 2; d.heads = 2; d.head_dim = 16;
  d.dense_ffn = 64; d.routed_experts = 8; d.experts_per_token = 3;
  d.shared_experts = 1; d.expert_ffn = 32; d.index_dim = 16;
  return d;
}
TableWeights weights_for(const DecoderDims& d) {
  ModelDims md; md.decoder = d;
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0.0f, 0.2f);
  TableWeights weights;
  for (const auto& spec : unlimited_ocr2_tensors(md)) {
    if (!spec.name.starts_with("model.layers") && !spec.name.starts_with("model.embed") &&
        spec.name != "model.norm.weight" && spec.name != "lm_head.weight") continue;
    Tensor tensor(spec.shape);
    const bool norm = spec.name.ends_with("norm.weight") && spec.name.find("router") == std::string::npos;
    for (float& v : tensor.span()) v = bf16_to_f32(f32_to_bf16(norm ? 1.0f : normal(rng)));
    weights.put(spec.name, std::move(tensor));
  }
  return weights;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  require(actual.size() == expected.size(), "recovered logits size");
  double error = 0, scale = 1e-12;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "finite recovered logits");
    error = std::max(error, std::abs(static_cast<double>(actual[i]) - expected[i]));
    scale = std::max(scale, std::abs(static_cast<double>(expected[i])));
  }
  require(error / scale < 2e-2, "recovered logits match reference without restart");
}
void reset_and_check(metal::MetalDecoder& decoder, const metal::Context& ctx, std::size_t weights_bytes) {
  decoder.reset_request_state();
  decoder.reset_request_state();
  require(metal::MetalDecoderTestAccess::cleared(decoder), "reset clears all request cache and metadata");
  require(ctx.allocated_bytes() == weights_bytes, "reset retains only model weights");
  require(ctx.impl().command == nil && ctx.impl().retired.empty(), "reset drains pending retired users");
}
}  // namespace

int main() {
  if (metal::device_count() == 0) return skip("no Metal device");
  reference::set_thread_count(2);
  return run_tests([] {
    const auto dims = tiny();
    const auto weights = weights_for(dims);
    auto router = Decoder::default_router(); router.block_size = 4;
    const Decoder ref_model(weights, dims, router);
    ReferenceDecoder ref(ref_model);
    const metal::Context ctx(0, 64ULL << 20);
    metal::MetalDecoder decoder(ctx, weights, dims, router);
    const auto weights_bytes = ctx.allocated_bytes();
    const auto small = build_prompt({12}), large = build_prompt({1024});
    const auto small_layout = layout_prompt(small), large_layout = layout_prompt(large);
    const auto small_embeds = ref.embed(small), large_embeds = ref.embed(large);
    const auto expected = ref.prefill(small_embeds, small_layout);
    const auto recover = [&] { compare(decoder.prefill(small_embeds, small_layout), expected); };
    recover();
    const auto old_capacity = metal::MetalDecoderTestAccess::capacity(decoder);

    // Four prefill arrays, then the first grown K succeeds. Fail V after
    // K has replaced its old allocation, before capacity_ is committed.
    ctx.impl().test_allocations_before_failure = 5;
    bool injected = false;
    try { decoder.prefill(large_embeds, large_layout); } catch (const std::bad_alloc&) { injected = true; }
    require(injected && metal::MetalDecoderTestAccess::partial_growth(decoder, old_capacity),
            "injected failure occurred after exactly the first cache replacement");
    reset_and_check(decoder, ctx, weights_bytes);
    recover();
    std::println("partial cache growth injection/reset/reuse PASS");

    // A real aggregate cap, without injection, also allows one replacement
    // but cannot retain the next one even after retiring the old K.
    const auto new_capacity = old_capacity * 2 + 1024;
    const auto rows = static_cast<std::int64_t>(large.size());
    const auto metadata_and_input = rows * (dims.hidden * 4 + 8) + ((rows + 3) / 4) * 16;
    ctx.impl().cap = ctx.allocated_bytes() + static_cast<std::size_t>(metadata_and_input +
        new_capacity * dims.heads * dims.head_dim * 4 + 4096);
    bool capped = false;
    try { decoder.prefill(large_embeds, large_layout); } catch (const std::length_error&) { capped = true; }
    require(capped && metal::MetalDecoderTestAccess::partial_growth(decoder, old_capacity),
            "real cap failed after first cache replacement");
    reset_and_check(decoder, ctx, weights_bytes);
    recover();  // Same reduced cap, same decoder; no model restart.
    std::println("real cap partial growth/reset/small reuse PASS");
    reset_and_check(decoder, ctx, weights_bytes);
    ctx.impl().cap = 64ULL << 20;

    // All request cache arrays succeed, then the first post-reserve scratch
    // allocation fails. This must release completed growth too.
    ctx.impl().test_allocations_before_failure = 4 + 4 * dims.layers;
    injected = false;
    try { decoder.prefill(small_embeds, small_layout); } catch (const std::bad_alloc&) { injected = true; }
    require(injected && metal::MetalDecoderTestAccess::capacity(decoder) > 0,
            "scratch failure followed completed reserve");
    reset_and_check(decoder, ctx, weights_bytes);
    recover();
    std::println("post-reserve scratch failure/reset/reuse PASS");

    // Exercise the parent's actual generation exception guard with native
    // allocations, rather than relying solely on an explicit reset call.
    ctx.impl().test_allocations_before_failure = 5;
    GenerationConfig config; config.max_new_tokens = 1;
    injected = false;
    try { generate_from_embeddings(decoder, large, large_embeds, config); }
    catch (const std::bad_alloc&) { injected = true; }
    require(injected && metal::MetalDecoderTestAccess::cleared(decoder), "generation guard invokes native reset");
    require(ctx.allocated_bytes() == weights_bytes, "generation guard returns retained budget");
    recover();
    reset_and_check(decoder, ctx, weights_bytes);
    std::println("native generation exception guard/small reuse PASS");
  });
}
