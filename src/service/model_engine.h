#pragma once

// The real engine behind the service: the upstream weights and tokenizer
// loaded from the models directory, the vision encoder and decoder on the
// configured backend, and the native generation loop.

#include <memory>

#include "service/config.h"
#include "service/engine.h"

namespace uocr::service {

// Loads the model as `config` says. Throws with the reason when the
// weights are missing or do not match the architecture, or the backend is
// not built into this binary.
std::unique_ptr<Engine> load_model_engine(const Config& config);

}  // namespace uocr::service
