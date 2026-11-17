#include "backend/metal/ops.h"
#include "backend/metal/context_impl.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdexcept>

namespace uocr::metal {
namespace {
Params params(std::initializer_list<std::int64_t> ints, std::initializer_list<float> floats = {}) {
  Params p; std::copy(ints.begin(), ints.end(), p.i); std::copy(floats.begin(), floats.end(), p.f); return p;
}
MPSMatrix* matrix(const Context& ctx, const float* ptr, std::int64_t rows, std::int64_t cols, std::int64_t ld) {
  const auto b = ctx.impl().binding(ptr);
  auto desc = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
    columns:static_cast<NSUInteger>(cols) rowBytes:static_cast<NSUInteger>(ld)*4 dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:b.buffer offset:b.offset descriptor:desc];
}
void product(const Context& ctx, bool transpose, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
             const float* a, std::int64_t lda, std::int64_t sa, const float* b, std::int64_t ldb, std::int64_t sb,
             float* c, std::int64_t ldc, std::int64_t sc, std::int64_t batch) {
  if(m<=0||n<=0||k<=0||batch<=0)throw std::invalid_argument("Metal GEMM shape");
  @autoreleasepool {
    auto& impl=ctx.impl();
    auto mul=[[MPSMatrixMultiplication alloc] initWithDevice:impl.device transposeLeft:NO transposeRight:transpose
      resultRows:static_cast<NSUInteger>(m) resultColumns:static_cast<NSUInteger>(n)
      interiorColumns:static_cast<NSUInteger>(k) alpha:alpha beta:0];
    for(std::int64_t j=0;j<batch;++j) {
      [mul encodeToCommandBuffer:impl.commands() leftMatrix:matrix(ctx,a+j*sa,m,k,lda)
        rightMatrix:matrix(ctx,b+j*sb,transpose?n:k,transpose?k:n,ldb) resultMatrix:matrix(ctx,c+j*sc,m,n,ldc)];
      ++impl.gemms;
    }
    if(++impl.pending>=64)impl.finish();
  }
}
}
void gemm_nt(const Context& c,std::int64_t m,std::int64_t n,std::int64_t k,float alpha,const float* a,std::int64_t lda,
 std::int64_t sa,const float* b,std::int64_t ldb,std::int64_t sb,float* out,std::int64_t ldc,std::int64_t sc,std::int64_t batch) {
 product(c,true,m,n,k,alpha,a,lda,sa,b,ldb,sb,out,ldc,sc,batch);
}
void gemm_nn(const Context& c,std::int64_t m,std::int64_t n,std::int64_t k,const float* a,std::int64_t lda,
 std::int64_t sa,const float* b,std::int64_t ldb,std::int64_t sb,float* out,std::int64_t ldc,std::int64_t sc,std::int64_t batch) {
 product(c,false,m,n,k,1,a,lda,sa,b,ldb,sb,out,ldc,sc,batch);
}
void linear(const Context& c,const float* x,std::int64_t m,std::int64_t k,const std::uint16_t* w,std::int64_t n,const float* bias,float* y) {
 if(m==1) dispatch(c,"gemv",n*32,{x,w,y},params({n,k}));
 else {
   Array b(c,DType::kF32,n*k);
   dispatch(c,"widen",n*k,{w,b.f32()},params({n*k}));
   gemm_nt(c,m,n,k,1,x,k,0,b.f32(),k,0,y,n,0,1);
 }
 if(bias)add_rows(c,y,bias,m,n);
}
void add_inplace(const Context& c,float* y,const float* x,std::int64_t n){dispatch(c,"element",n,{y,x},params({n,0}));}
void add_rows(const Context& c,float* y,const float* x,std::int64_t r,std::int64_t n){dispatch(c,"element",r*n,{y,x},params({r*n,1,n}));}
void layer_norm(const Context& c,float* x,const float* w,const float* b,std::int64_t r,std::int64_t n,float e){dispatch(c,"norm",r,{x,w,b},params({r,n,0},{e}));}
void rms_norm(const Context& c,float* x,const float* w,std::int64_t r,std::int64_t n,float e){dispatch(c,"norm",r,{x,w,nullptr},params({r,n,1},{e}));}
void router_norm(const Context& c,float* x,const float* w,std::int64_t r,std::int64_t n,float e){dispatch(c,"norm",r,{x,w,nullptr},params({r,n,2},{e}));}
void gelu_erf(const Context& c,float* x,std::int64_t n){dispatch(c,"element",n,{x,nullptr},params({n,5}));}
void quick_gelu(const Context& c,float* x,std::int64_t n){dispatch(c,"element",n,{x,nullptr},params({n,2}));}
void silu_mul(const Context& c,float* g,const float* u,std::int64_t n){dispatch(c,"element",n,{g,u},params({n,3}));}
void softmax_rows(const Context& c,float* x,std::int64_t r,std::int64_t n){dispatch(c,"softmax",r,{x},params({r,n}));}
void scale_add_mask(const Context& c,float* x,float s,const float* mask,std::int64_t b,std::int64_t r,std::int64_t n){dispatch(c,"element",b*r*n,{x,mask},params({b*r*n,6,r*n,mask!=nullptr},{s}));}
void rope(const Context& c,float* x,std::int64_t r,std::int64_t h,std::int64_t d,const std::int64_t* t,const std::int64_t* pg,
 const float* it,std::int64_t nt,const float* ip,std::int64_t np,std::int64_t mod,std::int64_t one_t,std::int64_t one_p){
 if(d%2||nt+np!=d/2)throw std::invalid_argument("Metal rope shape");
 dispatch(c,"rope_op",r*h*(d/2),{x,t,pg,it,ip},params({r,h,d,nt,np,mod,t!=nullptr,one_t,one_p}));
}
void im2col(const Context& c,const float* x,std::int64_t ch,std::int64_t h,std::int64_t w,std::int64_t kh,std::int64_t kw,int st,int pad,float* y){
 auto oh=(h+2*pad-kh)/st+1,ow=(w+2*pad-kw)/st+1;
 dispatch(c,"patches",oh*ow*ch*kh*kw,{x,y},params({ch,h,w,kh,kw,st,pad,oh,ow}));
}
void chw_to_hwc(const Context& c,const float* x,std::int64_t ch,std::int64_t h,std::int64_t w,float* y){dispatch(c,"transpose_op",ch*h*w,{x,y},params({ch,h*w}));}
void hwc_to_chw(const Context& c,const float* x,std::int64_t h,std::int64_t w,std::int64_t ch,float* y){dispatch(c,"transpose_op",ch*h*w,{x,y},params({h*w,ch}));}
void gather_rows(const Context& c,const float* x,const std::int64_t* ids,std::int64_t r,std::int64_t n,float* y){dispatch(c,"gather_op",r*n,{x,ids,y},params({r,n}));}
void gather_rows_bf16(const Context& c,const std::uint16_t* x,const std::int64_t* ids,std::int64_t r,std::int64_t n,float* y){dispatch(c,"gather_bf",r*n,{x,ids,y},params({r,n,1}));}
void gather_row_bf16(const Context& c,const std::uint16_t* x,std::int64_t id,std::int64_t n,float* y){dispatch(c,"gather_bf",n,{x,nullptr,y},params({1,n,0,id}));}
void scatter_add_rows(const Context& c,float* y,const std::int64_t* ids,const float* w,const float* x,std::int64_t r,std::int64_t n){dispatch(c,"scatter_op",n,{y,ids,w,x},params({r,n}));}
void concat_cols(const Context& c,const float* a,std::int64_t ca,const float* b,std::int64_t cb,std::int64_t r,float* y){dispatch(c,"concat_op",r*(ca+cb),{a,b,y},params({r,ca,cb}));}
void copy(const Context& c,float* y,const float* x,std::int64_t n){dispatch(c,"element",n,{y,x},params({n,4}));}
void rel_pos_bias(const Context& c,const float* q,std::int64_t ld,std::int64_t h,std::int64_t w,std::int64_t d,const float* rh,const float* rw,float* out){
 Array dots(c,DType::kF32,h*w*(h+w));
 dispatch(c,"rel_dot",h*w*(h+w),{q,rh,rw,dots.f32()},params({ld,h,w,d}));
 dispatch(c,"rel_expand",h*w*h*w,{dots.f32(),out},params({h,w}));
}
void block_summaries(const Context& c,const float* keys,std::int64_t s,std::int64_t d,const float* hc,std::int64_t bs,float* out){
 if(bs<1||bs>64)throw std::invalid_argument("Metal block size must be 1..64");
 dispatch(c,"summaries",(s+bs-1)/bs,{keys,hc,out},params({s,d,bs}));
}
void expert_topk(const Context& c,const float* x,std::int64_t r,std::int64_t e,std::int64_t k,std::int64_t* ids,float* weights){
 if(k<1||k>e)throw std::invalid_argument("Metal expert topk");
 dispatch(c,"topk",r,{x,ids,weights},params({r,e,k}));
}
void expert_ffn_one(const Context& c,const float* x,std::int64_t h,std::int64_t f,const std::uint16_t* g,const std::uint16_t* u,
 const std::uint16_t* down,const std::int64_t* ids,const float* weights,std::int64_t k,float* out){
 Array mid(c,DType::kF32,k*f);
 dispatch(c,"expert_up",k*f*32,{x,g,u,ids,mid.f32()},params({h,f,k}));
 dispatch(c,"expert_down",h*32,{mid.f32(),down,ids,weights,out},params({h,f,k}));
}
void route_blocks(const Context& c,const float* z,std::int64_t r,std::int64_t nb,std::int64_t pos,std::int64_t keys,
 const std::int64_t* lo,const std::int64_t* hi,std::int64_t prompts,const std::int64_t* pages,std::int64_t page,const RouteParams& p,float* mask){
 if(p.top_m>32||nb<1||p.sigma<=0)throw std::invalid_argument("Metal router parameters");
 dispatch(c,"route",r,{z,lo,hi,pages,mask},params({r,nb,pos,keys,prompts,page,pages!=nullptr,p.block_size,p.local_blocks,p.top_m,p.top_k},{p.idx_scale,p.floor,p.sigma}));
}
}  // namespace uocr::metal
namespace uocr::metal {
void resize_positions(const Context& c,const float* x,std::int64_t src,std::int64_t dst,std::int64_t width,float* y){
 dispatch(c,"resize_positions",dst*dst*width,{x,y},params({src,dst,width}));
}
void relative_table(const Context& c,const float* x,std::int64_t in,std::int64_t span,std::int64_t width,float* y){
 dispatch(c,"relative_table",span*span*width,{x,y},params({in,span,width}));
}
}
