// Qualification preflight only: query Level Zero Sysman without model work.
#include <level_zero/zes_api.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void checked(ze_result_t result) {
  if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("Level Zero Sysman query failed");
}
}  // namespace

int main() {
  try {
    checked(zesInit(0));
    std::uint32_t count = 0;
    checked(zesDriverGet(&count, nullptr));
    std::vector<zes_driver_handle_t> drivers(count);
    checked(zesDriverGet(&count, drivers.data()));
    std::uint32_t devices_checked = 0;
    for (const auto driver : drivers) {
      std::uint32_t ndevices = 0;
      checked(zesDeviceGet(driver, &ndevices, nullptr));
      std::vector<zes_device_handle_t> devices(ndevices);
      checked(zesDeviceGet(driver, &ndevices, devices.data()));
      for (const auto device : devices) {
        zes_device_properties_t properties{};
        properties.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        checked(zesDeviceGetProperties(device, &properties));
        std::printf("sysman_device=%s\n", properties.modelName);
        std::uint32_t nmodules = 0;
        checked(zesDeviceEnumMemoryModules(device, &nmodules, nullptr));
        std::vector<zes_mem_handle_t> modules(nmodules);
        checked(zesDeviceEnumMemoryModules(device, &nmodules, modules.data()));
        if (modules.empty()) throw std::runtime_error("no device memory telemetry");
        for (const auto module : modules) {
          zes_mem_state_t state{};
          state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
          checked(zesMemoryGetState(module, &state));
          std::printf("memory_free_bytes=%llu memory_size_bytes=%llu required_free_bytes=%llu\n",
                      static_cast<unsigned long long>(state.free), static_cast<unsigned long long>(state.size),
                      12ULL << 30);
          if (state.free < (12ULL << 30)) throw std::runtime_error("less than 12 GiB free GPU memory");
        }
        ++devices_checked;
      }
    }
    // This qualification host has one GPU. Refuse ambiguous telemetry on other hosts.
    if (devices_checked != 1) throw std::runtime_error("qualification requires exactly one Sysman GPU");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
