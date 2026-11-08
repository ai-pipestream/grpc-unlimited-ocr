#include "model/metal/device_weight.h"
#include "backend/metal/ops.h"
#include <stdexcept>
namespace uocr::metal {
DeviceWeight device_weight(const Context& c,const Tensor& t,WeightFormat) {
 return {upload_tensor(c,t,DType::kBF16),t.dim(0),t.numel()/t.dim(0)};
}
DeviceWeight stacked_weight(const Context& c,const WeightSource& w,const std::vector<std::string>& names,WeightFormat) {
 if(names.empty())throw std::invalid_argument("empty expert weights");
 const Tensor first=w.get(names.front());
 const auto rows=first.dim(0),cols=first.numel()/rows;
 DeviceWeight out{Array(c,DType::kBF16,static_cast<std::int64_t>(names.size())*rows*cols),static_cast<std::int64_t>(names.size())*rows,cols};
 for(std::size_t j=0;j<names.size();++j){
   const Tensor t=j==0?first:w.get(names[j]);
   if(t.shape()!=first.shape())throw std::invalid_argument("inconsistent expert shape");
   out.data.upload(t.data(),t.numel(),static_cast<std::int64_t>(j)*rows*cols);
 }
 return out;
}
void apply_weight(const Context& c,const float* x,std::int64_t m,const DeviceWeight& w,const float* bias,float* y,std::int64_t row,std::int64_t n){
 if(n<0)n=w.rows;
 if(row<0||n<=0||row>w.rows||n>w.rows-row)throw std::out_of_range("weight row block");
 linear(c,x,m,w.cols,w.bf16_at(row),n,bias,y);
}
}
