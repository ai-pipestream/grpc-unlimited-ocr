// The safetensors reader: a well-formed file maps and every tensor views
// the right bytes; each kind of malformed header is refused.

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "weights/safetensors.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

std::string file_bytes(const std::string& header, const std::string& data) {
  std::string out(8, '\0');
  const std::uint64_t size = header.size();
  std::memcpy(out.data(), &size, 8);
  return out + header + data;
}

void write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream(path, std::ios::binary) << bytes;
}

void parse(const std::string& header, std::size_t data_size) {
  std::vector<std::byte> data(data_size);
  parse_safetensors_header(header, data.data(), data.size());
}

void well_formed_file(const std::filesystem::path& dir) {
  // a: 2x2 BF16 (8 bytes), b: 3 F32 (12 bytes), stored b then a.
  const std::string header =
      R"({"__metadata__":{"format":"pt"},)"
      R"("a":{"dtype":"BF16","shape":[2,2],"data_offsets":[12,20]},)"
      R"("b":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})";
  std::string data(20, '\0');
  for (int i = 0; i < 20; ++i) {
    data[i] = static_cast<char>(i);
  }
  const auto path = dir / "one.safetensors";
  write(path, file_bytes(header, data));

  SafetensorsFile file(path);
  require(file.tensors().size() == 2, "two tensors, metadata skipped");
  const TensorInfo* a = file.find("a");
  require(a != nullptr && a->dtype == DType::kBF16 && a->shape == std::vector<std::int64_t>{2, 2}, "a's header");
  require(a->bytes.size() == 8 && static_cast<int>(a->bytes[0]) == 12, "a views its own bytes");
  require(file.find("b")->element_count() == 3, "b's element count");
  require(file.find("c") == nullptr, "absent name");

  SafetensorsFile moved(std::move(file));
  require(moved.find("a")->bytes.data() == a->bytes.data(), "a move keeps the mapping");
}

void malformed_headers() {
  require_throws([] { parse("[]", 0); }, "header not an object");
  require_throws([] { parse("{not json", 0); }, "header not JSON");
  require_throws([] { parse(R"({"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}})", 8); }, "unsupported dtype");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", 4); }, "negative dim");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", 4); }, "size mismatch");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 8); }, "trailing gap");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8); }, "leading gap");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", 2); }, "out of bounds");
  require_throws(
      [] {
        parse(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
              R"("b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8);
      },
      "overlap");
  require_throws([] { parse(R"({"a":{"dtype":"F32","shape":[1]}})", 4); }, "missing data_offsets");
  require_throws([] { parse(R"({"a":{"dtype":"U8","shape":[4294967296,4294967296,16],"data_offsets":[0,0]}})", 0); },
                 "shape overflow");
}

void malformed_files(const std::filesystem::path& dir) {
  write(dir / "short.safetensors", "abc");
  require_throws([&] { SafetensorsFile f(dir / "short.safetensors"); }, "file shorter than the prefix");
  std::string huge(8, '\0');
  const std::uint64_t size = 1ULL << 40;
  std::memcpy(huge.data(), &size, 8);
  write(dir / "huge.safetensors", huge + "{}");
  require_throws([&] { SafetensorsFile f(dir / "huge.safetensors"); }, "header length past the file");
  require_throws([&] { SafetensorsFile f(dir / "absent.safetensors"); }, "missing file");
}

void sharded(const std::filesystem::path& dir) {
  write(dir / "s1.safetensors", file_bytes(R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})", "ab"));
  write(dir / "s2.safetensors", file_bytes(R"({"y":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", "c"));
  write(dir / "index.json", R"({"weight_map":{"x":"s1.safetensors","y":"s2.safetensors"}})");
  const auto weights = ShardedWeights::open(dir / "index.json");
  require(weights.tensor_count() == 2 && weights.at("y").bytes.size() == 1, "two shards resolve");
  require_throws([&] { (void)weights.at("z"); }, "absent tensor");

  write(dir / "wrong.json", R"({"weight_map":{"x":"s2.safetensors","y":"s2.safetensors"}})");
  require_throws([&] { ShardedWeights::open(dir / "wrong.json"); }, "tensor in a different shard than indexed");
  write(dir / "unlisted.json", R"({"weight_map":{"x":"s1.safetensors"}})");
  ShardedWeights::open(dir / "unlisted.json");  // s2 is never opened, so nothing unlisted is seen
  write(dir / "extra.json", R"({"weight_map":{"x":"s1.safetensors","y":"s2.safetensors","w":"s2.safetensors"}})");
  require_throws([&] { ShardedWeights::open(dir / "extra.json"); }, "indexed tensor missing from its shard");
  write(dir / "escape.json", R"({"weight_map":{"x":"../s1.safetensors"}})");
  require_throws([&] { ShardedWeights::open(dir / "escape.json"); }, "shard path escapes the directory");
}

}  // namespace

int main() {
  return run_tests([] {
    const auto dir = test_dir("safetensors");
    well_formed_file(dir);
    malformed_headers();
    malformed_files(dir);
    sharded(dir);
  });
}
