#pragma once
#include "model/metal/device_params.h"
#include <vector>
namespace uocr::metal {
enum class WeightFormat { kBf16 };
struct DeviceWeight {
  Array data;
  std::int64_t rows=0, cols=0;
  const std::uint16_t* bf16_at(std::int64_t row) const {return data.bf16()+row*cols;}
};
DeviceWeight device_weight(const Context&,const Tensor&,WeightFormat);
DeviceWeight stacked_weight(const Context&,const WeightSource&,const std::vector<std::string>&,WeightFormat);
void apply_weight(const Context&,const float*,std::int64_t,const DeviceWeight&,const float*,float*,std::int64_t row=0,std::int64_t n=-1);
}
