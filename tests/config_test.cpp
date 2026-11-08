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
    require(defaults.transport_idle_ms == 30000, "transport idle default");
    require(defaults.device_memory_cap == 0, "device budget defaults to backend capacity policy");
    ::setenv("GRPC_UOCR_BACKEND", "intel-gpu", 1);
    ::setenv("GRPC_UOCR_DEVICE_MEMORY_CAP", "12884901888", 1);
    require(load_config_from_env().device_memory_cap == (12ULL << 30), "device budget retains 64-bit bytes");
    for (const auto* invalid : {"-1", "12GiB", "549755813889", "18446744073709551616"}) {
      ::setenv("GRPC_UOCR_DEVICE_MEMORY_CAP", invalid, 1);
      require_throws([] { load_config_from_env(); }, "invalid device budget rejected");
    }
    ::unsetenv("GRPC_UOCR_DEVICE_MEMORY_CAP");
    ::unsetenv("GRPC_UOCR_BACKEND");
    ::setenv("GRPC_UOCR_TRANSPORT_IDLE_MS", "150", 1);
    require(load_config_from_env().transport_idle_ms == 150, "transport idle override");
    ::setenv("GRPC_UOCR_TRANSPORT_IDLE_MS", "0", 1);
    require_throws([] { load_config_from_env(); }, "transport idle cannot disable the bound");
    ::unsetenv("GRPC_UOCR_TRANSPORT_IDLE_MS");

    ::setenv("GRPC_UOCR_BACKEND", "reference", 1);
    ::setenv("GRPC_UOCR_PAGE_WINDOW", "4", 1);
    ::setenv("GRPC_UOCR_MAX_PAGE_BYTES", "4096", 1);
    const auto set = load_config_from_env();
    require(set.backend == "reference" && set.page_window == 4, "overrides");
    require(set.weight_format == "fp32", "reference reports actual fp32 precision");
    ::setenv("GRPC_UOCR_WEIGHT_FORMAT", "int8", 1);
    require_throws([] { load_config_from_env(); }, "reference refuses ignored quantization");
    ::unsetenv("GRPC_UOCR_WEIGHT_FORMAT");
    require(set.max_stream_buffered_bytes == 4 * 4096, "the stream budget follows the page cap");

    ::setenv("GRPC_UOCR_DEVICE_MEMORY_CAP", "1048576", 1);
    for (const char* backend : {"metal", "intel-gpu"}) {
      ::setenv("GRPC_UOCR_BACKEND", backend, 1);
      require(load_config_from_env().device_memory_cap == 1048576, "device buffer cap retained");
    }
    for (const char* backend : {"cuda", "cpu", "reference"}) {
      ::setenv("GRPC_UOCR_BACKEND", backend, 1);
      require_throws([] { load_config_from_env(); }, "unsupported device cap must not be ignored");
    }
    ::unsetenv("GRPC_UOCR_DEVICE_MEMORY_CAP");

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
