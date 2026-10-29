#pragma once

// Plain-main test helpers, the same shape as the sibling repos: an
// assertion that throws, and the CTest skip code.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <stdexcept>
#include <string>

inline void require(bool condition, const std::string& what) {
  if (!condition) {
    throw std::runtime_error("FAIL: " + what);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& what) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("FAIL (no throw): " + what);
}

inline const char* env_or_null(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? value : nullptr;
}

// Exit 77 so CTest reports SKIP, not PASS.
inline int skip(const std::string& why) {
  std::println(stderr, "SKIP: {}", why);
  return 77;
}

// A per-test scratch directory under the build tree.
inline std::filesystem::path test_dir(const std::string& name) {
  const char* root = env_or_null("UOCR_TEST_TMP");
  auto dir = std::filesystem::path(root != nullptr ? root : "test-tmp") / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

template <typename Fn>
int run_tests(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& error) {
    std::println(stderr, "{}", error.what());
    return 1;
  }
  std::println("ok");
  return 0;
}
