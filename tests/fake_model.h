#pragma once

// Test doubles for the output and service tests: a deterministic
// tokenisation of model output text (layout control tokens whole, the rest
// in short byte runs that may split a UTF-8 character, as byte-level BPE
// does), and a scripted engine that "decodes" given text through the
// Engine interface with no weights and no device.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "service/engine.h"

namespace uocr::testing {

constexpr std::uint32_t kPageToken = 128827;
constexpr std::uint32_t kEosToken = 1;
constexpr std::string_view kEosText = "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";

// Splits `text` into token pieces: <PAGE>, <BLOCK>, <FORK>, <MERGE> and
// <LEVELn> whole, everything else in runs of 1 to `max_run` bytes chosen by
// `seed`.
inline std::vector<std::string> tokenize(std::string_view text, unsigned seed = 1, int max_run = 4) {
  static constexpr std::string_view kSpecials[] = {"<PAGE>", "<BLOCK>", "<FORK>", "<MERGE>", "<LEVEL"};
  std::mt19937 rng(seed);
  std::vector<std::string> out;
  std::size_t p = 0;
  while (p < text.size()) {
    bool special = false;
    for (const std::string_view s : kSpecials) {
      if (!text.substr(p).starts_with(s)) continue;
      std::size_t end = p + s.size();
      if (s == "<LEVEL") {
        while (end < text.size() && text[end] >= '0' && text[end] <= '9') ++end;
        if (end >= text.size() || text[end] != '>') break;
        ++end;
      }
      out.emplace_back(text.substr(p, end - p));
      p = end;
      special = true;
      break;
    }
    if (special) continue;
    std::size_t run = 1 + rng() % static_cast<unsigned>(max_run);
    std::size_t end = p;
    while (end < text.size() && end - p < run && text[end] != '<') ++end;
    if (end == p) end = p + 1;
    out.emplace_back(text.substr(p, end - p));
    p = end;
  }
  return out;
}

// A vocabulary grown on demand: one id per distinct piece.
class Vocabulary {
 public:
  Vocabulary() {
    ids_["<PAGE>"] = kPageToken;
    pieces_[kPageToken] = "<PAGE>";
    ids_[std::string(kEosText)] = kEosToken;
    pieces_[kEosToken] = std::string(kEosText);
  }
  std::uint32_t id(const std::string& piece) {
    std::lock_guard lock(mutex_);
    const auto [it, added] = ids_.emplace(piece, next_);
    if (added) pieces_[next_++] = piece;
    return it->second;
  }
  std::string piece(std::uint32_t id) const {
    std::lock_guard lock(mutex_);
    const auto it = pieces_.find(id);
    return it == pieces_.end() ? std::string() : it->second;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::uint32_t> ids_;
  std::map<std::uint32_t, std::string> pieces_;
  std::uint32_t next_ = 1000;
};

// How one window decodes.
struct WindowScript {
  std::string output;          // the raw output, special tokens kept, EOS excluded
  bool eos = true;             // end with the EOS step
  int throw_after = -1;        // throw after this many steps
  int ban_from = -1;           // steps from here on are marked as repeat-ban overrides
  service::WindowFinish finish = service::WindowFinish::kEndOfSequence;
};

class FakeEngine final : public service::Engine {
 public:
  std::vector<WindowScript> windows;  // consumed in order, one per decode call
  std::chrono::microseconds step_delay{0};
  std::atomic<int> decodes{0};
  std::atomic<int> concurrent{0};
  std::atomic<int> max_concurrent{0};
  std::vector<std::size_t> window_sizes;  // pages each decode call got

  service::EngineInfo info() const override { return {"fake", "none", "bf16", "0"}; }
  std::string piece(std::uint32_t token) const override { return vocab.piece(token); }
  std::string bytes(std::uint32_t token) const override { return vocab.piece(token); }
  std::uint32_t page_token() const override { return kPageToken; }
  std::uint32_t eos_token() const override { return kEosToken; }

  // A raster is "PNG <width> <height>"; anything else is unreadable.
  service::PageSize probe(std::string_view raster) const override {
    unsigned w = 0;
    unsigned h = 0;
    if (!raster.starts_with("PNG ") || std::sscanf(std::string(raster).c_str(), "PNG %u %u", &w, &h) != 2) {
      throw service::ImageError("not an image");
    }
    return {w, h};
  }

  service::WindowResult decode(const service::WindowRequest& request,
                               const std::function<void(std::uint64_t)>& started,
                               const std::function<bool(const service::EngineStep&)>& step) override {
    const int now = ++concurrent;
    int seen = max_concurrent.load();
    while (now > seen && !max_concurrent.compare_exchange_weak(seen, now)) {}
    struct Leave {
      std::atomic<int>& n;
      ~Leave() { --n; }
    } leave{concurrent};
    WindowScript script;
    {
      std::lock_guard lock(mutex_);
      window_sizes.push_back(request.pages.size());
      script = windows.at(static_cast<std::size_t>(decodes++));
    }
    started(100 * request.pages.size());
    std::vector<std::uint32_t> ids;
    for (const std::string& p : tokenize(script.output, 7)) ids.push_back(vocab.id(p));
    if (script.eos) ids.push_back(kEosToken);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (static_cast<int>(i) == script.throw_after) throw std::runtime_error("scripted engine failure");
      if (step_delay.count() > 0) std::this_thread::sleep_for(step_delay);
      service::EngineStep s{ids[i], -0.5f, ids[i], -0.5f, service::StepRule::kNone};
      if (script.ban_from >= 0 && static_cast<int>(i) >= script.ban_from) {
        s.rule = service::StepRule::kRepeatBan;
        s.model_token = ids[i] + 1;
      }
      if (!step(s)) return {100 * request.pages.size(), service::WindowFinish::kStopped};
    }
    return {100 * request.pages.size(), script.finish};
  }

  mutable Vocabulary vocab;

 private:
  std::mutex mutex_;
};

}  // namespace uocr::testing
