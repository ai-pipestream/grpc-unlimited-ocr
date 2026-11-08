#include "check.h"
#include "runtime/generator.h"

using namespace uocr;

namespace {
class FailingDecoder final : public DecoderModel {
 public:
  enum class Failure { none, embed, prefill, step } failure = Failure::none;
  mutable bool poisoned = false;
  int resets = 0;
  void reset_request_state() noexcept override { poisoned = false; ++resets; }
  Tensor embed(const std::vector<std::int64_t>& ids) const override {
    if (failure == Failure::embed) fail();
    return Tensor({static_cast<std::int64_t>(ids.size()), 1});
  }
  std::vector<float> prefill(const Tensor&, const PromptLayout&) override {
    require(!poisoned, "previous failed request released its cache");
    if (failure == Failure::prefill) fail();
    std::vector<float> logits(128830, -10.0f);
    logits[failure == Failure::step ? 3 : 1] = 10.0f;
    return logits;
  }
  std::vector<float> step(std::int64_t, RopeCoord, std::int64_t, const BlockPages&) override {
    fail();
  }
  std::int64_t block_size() const override { return 4; }
 private:
  [[noreturn]] void fail() const { poisoned = true; throw std::bad_alloc(); }
};
class FixtureVision final : public VisionModel {
 public:
  bool failing = true;
  FixtureVision() { newline_ = Tensor({1}); separator_ = Tensor({1}); }
  Tensor encode_view(const Tensor&) const override {
    if (failing) throw std::bad_alloc();
    return Tensor({1, 1});
  }
};
}

int main() {
  return run_tests([] {
    FailingDecoder decoder;
    auto prompt = build_prompt({1});
    const Tensor embeds = decoder.embed(prompt);
    GenerationConfig config;
    config.max_new_tokens = 2;
    config.page_count_guard = false;
    auto recover = [&] {
      decoder.failure = FailingDecoder::Failure::none;
      const int resets = decoder.resets;
      const auto result = generate_from_embeddings(decoder, prompt, embeds, config);
      require(result.finish == FinishReason::kEndOfSequence, "same decoder recovers without restart");
      require(decoder.resets == resets, "successful generation retains reusable cache");
    };
    for (auto failure : {FailingDecoder::Failure::prefill, FailingDecoder::Failure::step}) {
      decoder.failure = failure;
      const int resets = decoder.resets;
      require_throws([&] { generate_from_embeddings(decoder, prompt, embeds, config); }, "injected allocation failure");
      require(decoder.resets == resets + 1 && !decoder.poisoned, "failed inference resets request state once");
      recover();
    }
    const int resets = decoder.resets;
    require_throws([&] {
      generate_from_embeddings(decoder, prompt, embeds, config, [](const GeneratedToken&) -> bool {
        throw std::runtime_error("output sink failed");
      });
    }, "output failure propagates");
    require(decoder.resets == resets + 1, "output failure releases request state");
    recover();
    decoder.poisoned = true;
    FixtureVision vision;
    const int before_vision = decoder.resets;
    require_throws([&] { generate(vision, decoder, {PageViews{}}, config); }, "vision failure propagates");
    require(!decoder.poisoned && decoder.resets == before_vision + 1,
            "vision failure releases prior retained decoder cache once");
    recover();
    vision.failing = false;
    for (auto failure : {FailingDecoder::Failure::embed, FailingDecoder::Failure::prefill,
                         FailingDecoder::Failure::step}) {
      decoder.failure = failure;
      const int before = decoder.resets;
      bool original = false;
      try { generate(vision, decoder, {PageViews{}}, config); }
      catch (const std::bad_alloc&) { original = true; }
      require(original, "original allocation exception survives delegated generation");
      require(decoder.resets == before + 1 && !decoder.poisoned, "delegation resets exactly once");
      recover();
    }
    const int before_callback = decoder.resets;
    bool original = false;
    try {
      generate_from_embeddings(decoder, prompt, embeds, config, {}, [](std::size_t) {
        throw std::runtime_error("prefill callback failed");
      });
    } catch (const std::runtime_error& error) { original = std::string(error.what()) == "prefill callback failed"; }
    require(original && decoder.resets == before_callback + 1, "prefill callback failure preserves cause and resets once");
    recover();
    decoder.failure = FailingDecoder::Failure::step;
    const int before_stop = decoder.resets;
    const auto stopped = generate_from_embeddings(decoder, prompt, embeds, config,
                                                  [](const GeneratedToken&) { return false; });
    require(stopped.tokens.size() == 1 && decoder.resets == before_stop,
            "cooperative stop retains reusable request cache without throwing");
    recover();
  });
}
