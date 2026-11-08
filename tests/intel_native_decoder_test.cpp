// The Intel SYCL decoder against the reference decoder on a tiny random model:
// logits after the prompt and after three decode steps. Parameters are on
// the bf16 grid; the remaining gap is bf16 rounding of GEMM inputs, so the
// bound is bf16-sized. Skips (77) without a Intel SYCL device.

#include <cmath>
#include <random>

#include "backend/intel/native/context.h"
#include "backend/reference/parallel.h"
#include "check.h"
#include "intel_numeric_checks.h"
#include "model/decoder.h"
#include "model/intel/device_weight.h"
#include "model/intel/intel_decoder.h"
#include "runtime/prompt_builder.h"
#include "tensor/dtype.h"
#include "weights/architecture.h"

using namespace uocr;

namespace {

DecoderDims tiny() {
  DecoderDims d;
  d.vocab = 128830;
  d.hidden = 32;
  d.layers = 2;
  d.heads = 2;
  d.head_dim = 16;
  d.dense_ffn = 64;  // widths are multiples of 32 so int8 weights apply
  d.routed_experts = 8;
  d.experts_per_token = 3;
  d.shared_experts = 1;
  d.expert_ffn = 32;
  d.index_dim = 16;  // equal to head_dim, as in the real model (they share a rope table)
  return d;
}

TableWeights weights_for(const DecoderDims& d) {
  ModelDims md;
  md.decoder = d;
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0.0f, 0.2f);
  TableWeights table;
  for (const auto& spec : unlimited_ocr2_tensors(md)) {
    if (!spec.name.starts_with("model.layers") && !spec.name.starts_with("model.embed") &&
        spec.name != "model.norm.weight" && spec.name != "lm_head.weight") {
      continue;
    }
    Tensor t(spec.shape);
    const bool norm = spec.name.ends_with("norm.weight") && spec.name.find("router") == std::string::npos;
    for (float& v : t.span()) v = bf16_to_f32(f32_to_bf16(norm ? 1.0f : normal(rng)));
    table.put(spec.name, std::move(t));
  }
  return table;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
  return intel_testing::finite_relative_error(a, b);
}

}  // namespace

void compare(intel_native::WeightFormat format, double tolerance) {
  std::println("weights {}", format == intel_native::WeightFormat::kBf16 ? "bf16" : "int8");
  const DecoderDims d = tiny();
  const auto w = weights_for(d);
  const auto reference_weights = weights_for(d);
  RouterParams rp = Decoder::default_router();
  rp.block_size = 4;
  rp.local_blocks = 1;
  rp.idx_scale = 4.0f;
  const Decoder ref_model(reference_weights, d, rp);
  ReferenceDecoder ref(ref_model);
  const intel_native::Context ctx(0);
  intel_native::IntelDecoder dev(ctx, w, d, rp, format);

  const auto prompt = build_prompt({12, 12});
  const Tensor embeds = ref.embed(prompt);
  const Tensor dev_embeds = dev.embed(prompt);
  require(rel(std::vector<float>(dev_embeds.span().begin(), dev_embeds.span().end()),
              std::vector<float>(embeds.span().begin(), embeds.span().end())) == 0.0,
          "embeddings are exact");
  const PromptLayout layout = layout_prompt(prompt);
  const double first = rel(dev.prefill(embeds, layout), ref.prefill(embeds, layout));
  std::println("prefill logits: relative error {:.2e}", first);
  require(first < tolerance, "prefill logits agree");

  DecodeState state(layout, prompt);
  const BlockPages blocks = block_image_pages(layout, rp.block_size);
  // A marker, then enough text that the decode attention reads remote
  // blocks as well as its local window.
  std::vector<std::int64_t> feed{7, 9, PromptTokens{}.page, 11};
  for (int i = 0; i < 40; ++i) feed.push_back(100 + 13 * i);
  for (const std::int64_t t : feed) {
    const RopeCoord c = state.advance(t);
    const double e = rel(dev.step(t, c, state.query_page(), blocks), ref.step(t, c, state.query_page(), blocks));
    if (t < 128 || t == PromptTokens{}.page) std::println("step {:>6}: relative error {:.2e}", t, e);
    require(e < tolerance, "decode logits agree");
  }
}

int main() {
  uocr::reference::set_thread_count(2);
  if (intel_native::device_count() == 0) return skip("no Intel SYCL device");
  return run_tests([] { compare(intel_native::WeightFormat::kBf16, 2e-2); });
}
