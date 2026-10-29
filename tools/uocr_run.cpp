// Developer runner: parses page images with the reference engine and
// streams the model's raw output to stdout. Not the service (that one is
// diskless and speaks gRPC); this reads its pages from files.
//
//   uocr-run --models DIR [--tiled] [--ngram N --window N] [--max-new N] [--save DIR] page1.png ...
//
// --save DIR keeps everything the run produced: the prompt ids, the raw
// output with every special token, and one row per decode step (token,
// piece, log-probability, the model's own top token, any override).

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <string>
#include <vector>

#include "image/page_preprocess.h"
#include "model/decoder.h"
#include "model/vision_encoder.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/architecture.h"
#include "weights/sharded_weights.h"

#if UOCR_WITH_CUDA
#include "backend/cuda/context.h"
#include "model/cuda/cuda_decoder.h"
#include "model/cuda/cuda_vision.h"
#endif

using namespace uocr;

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

const char* intervention_name(Intervention i) {
  switch (i) {
    case Intervention::kNone: return "none";
    case Intervention::kRepeatBan: return "repeat_ban";
    case Intervention::kPageGuard: return "page_guard";
  }
  return "unknown";
}

void save_run(const std::filesystem::path& dir, const Tokenizer& tok, const GenerationResult& r,
              const std::vector<std::string>& pages) {
  std::filesystem::create_directories(dir);
  std::ofstream prompt(dir / "prompt_ids.txt");
  for (const auto id : r.prompt) prompt << id << '\n';
  std::ofstream raw(dir / "output_raw.txt", std::ios::binary);
  raw << tok.decode(r.tokens);
  std::ofstream trace(dir / "trace.tsv");
  trace << "step\ttoken\tpiece\tlogprob\tmodel_token\tmodel_logprob\tintervention\n";
  for (std::size_t i = 0; i < r.trace.size(); ++i) {
    const auto& s = r.trace[i];
    trace << i << '\t' << s.token << '\t' << tok.piece(s.token) << '\t' << s.logprob << '\t' << s.model_token
          << '\t' << s.model_logprob << '\t' << intervention_name(s.intervention) << '\n';
  }
  std::ofstream meta(dir / "run.txt");
  meta << "finish\t" << static_cast<int>(r.finish) << "\nprompt_tokens\t" << r.prompt.size() << "\ngenerated_tokens\t"
       << r.tokens.size() << '\n';
  for (const auto& p : pages) meta << "page\t" << p << '\n';
}

int usage() {
  std::println(stderr, "usage: uocr-run --models DIR [--tiled] [--ngram N] [--window N] [--max-new N] [--save DIR] [--backend reference|cuda] page...");
  return 64;
}

}  // namespace

int main(int argc, char** argv) {
  std::string models;
  std::vector<std::string> pages;
  std::string save_dir;
  std::string backend = "reference";
  PreprocessOptions pre;
  GenerationConfig gen;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (a == "--models") models = value();
    else if (a == "--tiled") pre.tiled = true;
    else if (a == "--ngram") gen.ngram_size = std::stoll(value());
    else if (a == "--window") gen.ngram_window = std::stoll(value());
    else if (a == "--max-new") gen.max_new_tokens = std::stoll(value());
    else if (a == "--save") save_dir = value();
    else if (a == "--backend") backend = value();
    else if (a.starts_with("--")) return usage();
    else pages.push_back(a);
  }
  if (models.empty() || pages.empty()) return usage();
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const auto problems = check_architecture(unlimited_ocr2_tensors(), shards);
    if (!problems.empty()) {
      for (const auto& p : problems) std::println(stderr, "{}", p);
      return 1;
    }
    const Tokenizer tokenizer = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const CheckpointWeights weights(shards);
#if UOCR_WITH_CUDA
    std::unique_ptr<cuda::Context> cuda_ctx;  // declared first: device arrays must go before it
#endif
    std::unique_ptr<VisionModel> vision_model;
    std::unique_ptr<Decoder> reference_decoder;
    std::unique_ptr<DecoderModel> decoder_model;
    if (backend == "reference") {
      vision_model = std::make_unique<VisionEncoder>(weights);
      reference_decoder = std::make_unique<Decoder>(weights);
      decoder_model = std::make_unique<ReferenceDecoder>(*reference_decoder);
#if UOCR_WITH_CUDA
    } else if (backend == "cuda") {
      cuda_ctx = std::make_unique<cuda::Context>(0);
      vision_model = std::make_unique<cuda::CudaVisionEncoder>(*cuda_ctx, weights);
      decoder_model = std::make_unique<cuda::CudaDecoder>(*cuda_ctx, weights);
#endif
    } else {
      std::println(stderr, "unknown or unbuilt backend: {}", backend);
      return 64;
    }
    const VisionModel& vision = *vision_model;
    DecoderModel& decoder = *decoder_model;
    std::vector<PageViews> views;
    for (const auto& p : pages) views.push_back(preprocess_page(decode_image(read_file(p)), pre));
    const auto t1 = std::chrono::steady_clock::now();
    std::println(stderr, "loaded in {:.1f} s", std::chrono::duration<double>(t1 - t0).count());

    std::vector<std::int64_t> so_far;
    auto first = std::chrono::steady_clock::time_point{};
    const auto result = generate(vision, decoder, views, gen, [&](const GeneratedToken& step) {
      if (first == std::chrono::steady_clock::time_point{}) first = std::chrono::steady_clock::now();
      if (step.token == 1) return true;  // EOS: kept in the trace, not printed
      // Bytes go out as they come; a character split across tokens is
      // completed by the next write.
      so_far.push_back(step.token);
      std::fputs(tokenizer.decode({step.token}).c_str(), stdout);
      std::fflush(stdout);
      return true;
    });
    if (!save_dir.empty()) save_run(save_dir, tokenizer, result, pages);
    const auto t2 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t2 - first).count();
    std::println(stderr, "\nprompt {} tokens, prefill {:.1f} s, {} tokens in {:.1f} s ({:.2f} tok/s), finish {}",
                 result.prompt.size(), std::chrono::duration<double>(first - t1).count(), result.tokens.size(),
                 decode_s, static_cast<double>(result.tokens.size()) / std::max(decode_s, 1e-9),
                 static_cast<int>(result.finish));
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
  return 0;
}
