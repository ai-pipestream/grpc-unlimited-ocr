// GRPC_UOCR_* configuration: defaults, overrides and refusals.

#include <cstdlib>

#include "check.h"
#include "service/config.h"

using uocr::service::load_config_from_env;

int main() {
  return run_tests([] {
    const auto defaults = load_config_from_env();
    require(defaults.listen_address == "0.0.0.0:50073", "default listen address");
    require(defaults.page_window == 10 && defaults.ngram_size == 35, "upstream defaults");
    require(defaults.documents_in_flight == 1, "one document at a time by default");

    ::setenv("GRPC_UOCR_BACKEND", "reference", 1);
    ::setenv("GRPC_UOCR_PAGE_WINDOW", "4", 1);
    ::setenv("GRPC_UOCR_MAX_PAGE_BYTES", "4096", 1);
    const auto set = load_config_from_env();
    require(set.backend == "reference" && set.page_window == 4, "overrides");
    require(set.max_stream_buffered_bytes == 4 * 4096, "the stream budget follows the page cap");

    ::setenv("GRPC_UOCR_BACKEND", "tpu", 1);
    require_throws([] { load_config_from_env(); }, "an unknown backend is refused");
    ::setenv("GRPC_UOCR_BACKEND", "cuda", 1);
    ::setenv("GRPC_UOCR_PAGE_WINDOW", "0", 1);
    require_throws([] { load_config_from_env(); }, "a window of 0 is refused");
    ::setenv("GRPC_UOCR_PAGE_WINDOW", "-3", 1);
    require_throws([] { load_config_from_env(); }, "a negative number is refused");
    ::setenv("GRPC_UOCR_PAGE_WINDOW", "2x", 1);
    require_throws([] { load_config_from_env(); }, "trailing junk is refused");
  });
}
