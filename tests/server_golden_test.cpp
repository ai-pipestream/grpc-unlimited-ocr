// The golden page through the real server and the real weights. On the
// reference backend every step token must match the fp32 goldens. A fast
// backend is judged on structure and text, as the design says: the same
// blocks with the same kinds and text, boxes within 1/1000.
//
// Skips (77) unless UOCR_MODELS_DIR points at the weights. GRPC_UOCR_BACKEND
// picks the backend (default: cuda when built, else reference).

#include <cstdlib>
#include <sstream>

#include "check.h"
#include "output_helpers.h"
#include "service/model_engine.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

std::vector<std::uint32_t> golden_tokens(const std::string& path) {
  std::istringstream in(read_text(path));
  std::vector<std::uint32_t> out;
  for (std::uint32_t t = 0; in >> t;) out.push_back(t);
  return out;
}

}  // namespace

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR is not set");
  return run_tests([&] {
    ::setenv("GRPC_UOCR_MODELS_DIR", models, 1);
#if UOCR_WITH_CUDA
    ::setenv("GRPC_UOCR_BACKEND", "cuda", 0);
#else
    ::setenv("GRPC_UOCR_BACKEND", "reference", 0);
#endif
    ::setenv("GRPC_UOCR_NGRAM_SIZE", "0", 0);  // the goldens were recorded without the n-gram ban
    const service::Config config = service::load_config_from_env();
    const auto engine = service::load_model_engine(config);
    service::UnlimitedOcrServiceImpl service(config, *engine);
    const std::string dir = std::string(UOCR_GOLDENS) + "/quickstart-p2";
    auto requests = document(options_request(1));
    requests[1] = page_request(1, read_text(dir + "/page.png"));
    const Outcome out = run(service, requests);
    require(out.status.ok(), "the golden page parses: " + out.status.error_message());

    const bool exact = config.backend == "reference";
    const std::string raw = read_text(dir + "/output_raw.txt");
    const auto tokens = out.tokens();
    require(tokens.size() == 1, "one page of tokens");
    if (exact) {
      require(tokens[0].text() == raw + engine->bytes(engine->eos_token()), "the golden output, then EOS");
      const auto expected = golden_tokens(dir + "/tokens.txt");
      require(static_cast<std::size_t>(tokens[0].steps_size()) == expected.size() + 1, "every step, EOS included");
      for (std::size_t i = 0; i < expected.size(); ++i) {
        require(tokens[0].steps(static_cast<int>(i)).token() == expected[i], "step " + std::to_string(i));
      }
    }
    const auto blocks = out.blocks();
    const auto want = parse_whole(raw.substr(6));
    require(blocks.size() == want.size(), "as many blocks as the golden output");
    for (std::size_t i = 0; i < want.size(); ++i) {
      const auto& got = blocks[i];
      require(got.kind() == want[i].kind() && got.label_text() == want[i].label_text(), "block kind");
      require(got.text().text() == want[i].text().text(), "block text " + std::to_string(i));
      for (int c = 0; c < want[i].raw_box_size() && c < got.raw_box_size(); ++c) {
        require(std::abs(got.raw_box(c) - want[i].raw_box(c)) <= (exact ? 0 : 1), "box within 1/1000");
      }
    }
    std::println("{} blocks, {} steps on {}", blocks.size(), tokens[0].steps_size(), config.backend);
  });
}
