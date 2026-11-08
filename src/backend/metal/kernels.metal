#include <metal_stdlib>
using namespace metal;
struct Params { long i[20]; float f[4]; };
float bf(ushort v) { return as_type<float>(uint(v) << 16); }
kernel void widen(device const ushort* x [[buffer(0)]], device float* y [[buffer(1)]],
                  constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) { if(id < p.i[0]) y[id]=bf(x[id]); }
// One SIMD group owns an output row; adjacent lanes read adjacent bf16
// weights. All reduction arithmetic is float, including on pre-bf16 M2 GPUs.
kernel void gemv(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                 device float* y [[buffer(2)]], constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long row=id/32,lane=id%32;if(row>=p.i[0])return;
  float sum=0;for(long j=lane;j<p.i[1];j+=32)sum+=x[j]*bf(w[row*p.i[1]+j]);
  sum=simd_sum(sum);if(lane==0)y[row]=sum;
}
kernel void element(device float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                    constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  if(id>=p.i[0]) return;
  float v=y[id];
  switch(p.i[1]) {
    case 0: y[id]+=x[id]; break;
    case 1: y[id]+=x[id%p.i[2]]; break;
    case 2: y[id]=v/(1+exp(-1.702f*v)); break;
    case 3: y[id]=v/(1+exp(-v))*x[id]; break;
    case 4: y[id]=x[id]; break;
    case 5: {
      // Maximum absolute erf approximation error 1.5e-7.
      float z=abs(v)*0.7071067811865475f, t=1/(1+0.3275911f*z);
      float er=1-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-z*z);
      y[id]=0.5f*v*(1+copysign(er,v)); break;
    }
    case 6: y[id]=v*p.f[0]+(p.i[3]?x[id%p.i[2]]:0); break;
  }
}
kernel void norm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
                 constant Params& p [[buffer(8)]], uint row [[thread_position_in_grid]]) {
  if(row>=p.i[0]) return;
  long n=p.i[1], mode=p.i[2], off=long(row)*n;
  float sum=0,sq=0,anchor=mode==0?x[off]:0;
  for(long c=0;c<n;++c) {
    float v=x[off+c]; if(mode==2&&!isfinite(v)) { v=0; x[off+c]=0; }
    if(mode==0) sum+=v-anchor; else sq+=v*v;
  }
  float mean=mode==0?sum/float(n):0;
  // Center before squaring: E[x*x]-E[x]^2 loses low variance at large
  // finite offsets. Anchoring the mean also avoids summing that offset.
  if(mode==0) for(long c=0;c<n;++c) { float delta=(x[off+c]-anchor)-mean; sq+=delta*delta; }
  float inv=rsqrt(max(sq/float(n),0.0f)+p.f[0]);
  for(long c=0;c<n;++c) {
    float v=((x[off+c]-anchor)-mean)*inv;
    x[off+c]=mode==2?clamp(clamp(v,-1e4f,1e4f)*(1+w[c]),-1e4f,1e4f):v*w[c]+(mode==0?b[c]:0);
  }
}
kernel void softmax(device float* x [[buffer(0)]], constant Params& p [[buffer(8)]], uint row [[thread_position_in_grid]]) {
  if(row>=p.i[0]) return;
  long n=p.i[1], off=long(row)*n; float peak=-INFINITY;
  for(long j=0;j<n;++j) peak=max(peak,x[off+j]);
  float sum=0;
  for(long j=0;j<n;++j) { float e=peak==-INFINITY?0:exp(x[off+j]-peak); x[off+j]=e; sum+=e; }
  if(sum>0) for(long j=0;j<n;++j) x[off+j]/=sum;
}
kernel void rope_op(device float* x [[buffer(0)]], device const long* tok [[buffer(1)]], device const long* page [[buffer(2)]],
                    device const float* it [[buffer(3)]], device const float* ip [[buffer(4)]],
                    constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long half_dim=p.i[2]/2, heads=p.i[1]; if(id>=p.i[0]*heads*half_dim) return;
  long r=id/(heads*half_dim), h=(id/half_dim)%heads, j=id%half_dim;
  long tk=p.i[6]?tok[r]:p.i[7], pg=p.i[6]?page[r]:p.i[8];
  if(p.i[5]>0) pg=((pg%p.i[5])+p.i[5])%p.i[5];
  float angle=j<p.i[3]?float(tk)*it[j]:float(pg)*ip[j-p.i[3]], s=sin(angle), c=cos(angle);
  long off=(r*heads+h)*p.i[2]+j; float a=x[off],b=x[off+half_dim];
  x[off]=a*c-b*s; x[off+half_dim]=b*c+a*s;
}
kernel void patches(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant Params& p [[buffer(8)]],
                    uint id [[thread_position_in_grid]]) {
  long c=p.i[0],h=p.i[1],w=p.i[2],kh=p.i[3],kw=p.i[4],st=p.i[5],pad=p.i[6],oh=p.i[7],ow=p.i[8];
  long width=c*kh*kw; if(id>=oh*ow*width) return;
  long pos=id/width,e=id%width,iy=(pos/ow)*st-pad+(e/kw)%kh,ix=(pos%ow)*st-pad+e%kw;
  y[id]=iy>=0&&iy<h&&ix>=0&&ix<w?x[((e/(kh*kw))*h+iy)*w+ix]:0;
}
kernel void transpose_op(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant Params& p [[buffer(8)]],
                         uint id [[thread_position_in_grid]]) {
  if(id<p.i[0]*p.i[1]) y[(id%p.i[1])*p.i[0]+id/p.i[1]]=x[id];
}
kernel void gather_op(device const float* x [[buffer(0)]], device const long* ids [[buffer(1)]], device float* y [[buffer(2)]],
                      constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  if(id<p.i[0]*p.i[1]) y[id]=x[ids[id/p.i[1]]*p.i[1]+id%p.i[1]];
}
kernel void gather_bf(device const ushort* x [[buffer(0)]], device const long* ids [[buffer(1)]], device float* y [[buffer(2)]],
                      constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  if(id<p.i[0]*p.i[1]) y[id]=bf(x[(p.i[2]?ids[id/p.i[1]]:p.i[3])*p.i[1]+id%p.i[1]]);
}
kernel void scatter_op(device float* y [[buffer(0)]], device const long* ids [[buffer(1)]], device const float* w [[buffer(2)]],
                       device const float* x [[buffer(3)]], constant Params& p [[buffer(8)]], uint c [[thread_position_in_grid]]) {
  if(c>=p.i[1]) return;
  for(long r=0;r<p.i[0];++r) y[ids[r]*p.i[1]+c]+=w[r]*x[r*p.i[1]+c];
}
kernel void concat_op(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* y [[buffer(2)]],
                      constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long width=p.i[1]+p.i[2]; if(id>=p.i[0]*width) return;
  long r=id/width,c=id%width; y[id]=c<p.i[1]?a[r*p.i[1]+c]:b[r*p.i[2]+c-p.i[1]];
}
kernel void rel_dot(device const float* q [[buffer(0)]], device const float* rh [[buffer(1)]], device const float* rw [[buffer(2)]],
                    device float* out [[buffer(3)]], constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long h=p.i[1],w=p.i[2],d=p.i[3],span=h+w; if(id>=h*w*span)return;
  long t=id/span,j=id%span; float sum=0;
  for(long e=0;e<d;++e) sum+=q[t*p.i[0]+e]*(j<h?rh[((t/w)*h+j)*d+e]:rw[((t%w)*w+j-h)*d+e]);
  out[id]=sum;
}
kernel void rel_expand(device const float* dots [[buffer(0)]], device float* out [[buffer(1)]],
                       constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long h=p.i[0],w=p.i[1],n=h*w,span=h+w; if(id>=n*n)return;
  long t=id/n,k=id%n; out[id]=dots[t*span+k/w]+dots[t*span+h+k%w];
}
kernel void summaries(device const float* keys [[buffer(0)]], device const float* hc [[buffer(1)]], device float* out [[buffer(2)]],
                      constant Params& p [[buffer(8)]], uint blk [[thread_position_in_grid]]) {
  long s=p.i[0],d=p.i[1],bs=p.i[2],first=long(blk)*bs,last=min(first+bs,s); if(first>=s)return;
  float ws[64],peak=-INFINITY,total=0;
  for(long t=first;t<last;++t) { float v=0; for(long e=0;e<d;++e) v+=keys[t*d+e]*hc[e];
    v=clamp(v*rsqrt(float(d)),-1e4f,1e4f); ws[t-first]=v;peak=max(peak,v); }
  for(long t=first;t<last;++t){ws[t-first]=exp(ws[t-first]-peak);total+=ws[t-first];}
  for(long e=0;e<d;++e){float sum=0;for(long t=first;t<last;++t)sum+=ws[t-first]/max(total,1e-6f)*keys[t*d+e];out[long(blk)*d+e]=sum;}
}
kernel void topk(device const float* logits [[buffer(0)]], device long* ids [[buffer(1)]], device float* weights [[buffer(2)]],
                 constant Params& p [[buffer(8)]], uint row [[thread_position_in_grid]]) {
  if(row>=p.i[0])return;long n=p.i[1],k=p.i[2],off=long(row)*n; float peak=-INFINITY,total=0;
  for(long e=0;e<n;++e)peak=max(peak,logits[off+e]);
  for(long e=0;e<n;++e)total+=exp(logits[off+e]-peak);
  for(long j=0;j<k;++j){long best=-1;float val=-INFINITY;
    for(long e=0;e<n;++e){bool used=false;for(long t=0;t<j;++t)used|=ids[long(row)*k+t]==e;
      if(!used&&(best<0||logits[off+e]>val)){best=e;val=logits[off+e];}}
    ids[long(row)*k+j]=best;weights[long(row)*k+j]=exp(val-peak)/total;}
}
kernel void expert_up(device const float* x [[buffer(0)]], device const ushort* g [[buffer(1)]], device const ushort* u [[buffer(2)]],
                      device const long* ids [[buffer(3)]], device float* mid [[buffer(4)]],
                      constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long hidden=p.i[0],ffn=p.i[1],item=id/32,lane=id%32;if(item>=p.i[2]*ffn)return;
  long row=ids[item/ffn]*ffn+item%ffn;float a=0,b=0;
  for(long c=lane;c<hidden;c+=32){float v=x[c];a+=v*bf(g[row*hidden+c]);b+=v*bf(u[row*hidden+c]);}
  a=simd_sum(a);b=simd_sum(b);if(lane==0)mid[item]=a/(1+exp(-a))*b;
}
kernel void expert_down(device const float* mid [[buffer(0)]], device const ushort* down [[buffer(1)]], device const long* ids [[buffer(2)]],
                        device const float* weights [[buffer(3)]], device float* out [[buffer(4)]],
                        constant Params& p [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  long hidden=p.i[0],ffn=p.i[1],c=id/32,lane=id%32;if(c>=hidden)return;float sum=0;
  for(long j=0;j<p.i[2];++j){float v=0;long row=ids[j]*hidden+c;for(long e=lane;e<ffn;e+=32)v+=mid[j*ffn+e]*bf(down[row*ffn+e]);
    v=simd_sum(v);sum+=weights[j]*v;}
  if(lane==0)out[c]=sum;
}
bool candidate(long b,long remote,long pg,long prompts,device const long* lo,device const long* hi,bool restrict_page) {
  return b<=remote&&(!restrict_page||(pg>=0&&b<prompts&&lo[b]>=0&&lo[b]<=pg&&pg<=hi[b]));
}
kernel void route(device const float* z [[buffer(0)]],device const long* lo [[buffer(1)]],device const long* hi [[buffer(2)]],
                  device const long* qp [[buffer(3)]],device float* mask [[buffer(4)]],
                  constant Params& p [[buffer(8)]],uint row [[thread_position_in_grid]]) {
  if(row>=p.i[0])return;
  long nb=p.i[1],pos=p.i[2]+row,keys=p.i[3],prompts=p.i[4],pg=p.i[6]?qp[row]:p.i[5];
  long bs=p.i[7],local=p.i[8],remote=pos/bs-local;
  int m=int(min(max(p.i[9],1L),nb)),k=int(min(max(p.i[10],1L),long(m)));
  bool restrict_page=false;
  for(long b=0;b<nb;++b)restrict_page|=candidate(b,remote,pg,prompts,lo,hi,true);
  float top[32],tau[32],bias[32];long idx[32];bool valid[32];
  for(int j=0;j<m;++j){float best=-INFINITY;long bi=-1;
    for(long b=0;b<nb;++b){bool used=false;for(int t=0;t<j;++t)used|=idx[t]==b;if(used)continue;
      float v=candidate(b,remote,pg,prompts,lo,hi,restrict_page)?clamp(z[long(row)*nb+b],-1e4f,1e4f)*p.f[0]:-1e4f;
      if(bi<0||v>best){best=v;bi=b;}}
    top[j]=best;idx[j]=bi;valid[j]=candidate(bi,remote,pg,prompts,lo,hi,restrict_page);}
  float peak=top[0],sum=0,sq=0;int support=0;
  for(int j=0;j<m;++j){float v=(top[j]-peak)*0.5f;top[j]=v;sum+=v;sq+=v*v;
    float rho=float(j+1),mean=sum/rho,ss=rho*(sq/rho-mean*mean);
    tau[j]=mean-sqrt(max((1-ss)/rho,0.0f));if(tau[j]<=v)++support;}
  float threshold=tau[max(support,1)-1],mu=0;int count=0;
  for(int j=0;j<k;++j){float v=max(top[j]-threshold,0.0f);bias[j]=log(max(v*v,p.f[1]));if(valid[j]){mu+=bias[j];++count;}}
  mu/=float(max(count,1));
  for(long t=0;t<keys;++t){float v=-INFINITY;if(t<=pos){if(t>pos-local*bs)v=0;
    else for(int j=0;j<k;++j)if(valid[j]&&idx[j]==t/bs)v=(bias[j]-mu)/p.f[2];}
    mask[long(row)*keys+t]=v;}
}
float cubic(float x) {
  x=abs(x);if(x<1)return (1.5f*x-2.5f)*x*x+1;
  if(x<2)return ((-0.5f*x+2.5f)*x-4)*x+2;
  return 0;
}
// Position embeddings only: HWC, antialiased separable cubic, fp32.
kernel void resize_positions(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
 constant Params& p [[buffer(8)]],uint id [[thread_position_in_grid]]) {
 long src=p.i[0],dst=p.i[1],c=p.i[2];if(id>=dst*dst*c)return;
 long ch=id%c,ox=(id/c)%dst,oy=id/(c*dst);float scale=float(src)/float(dst),support=2*max(scale,1.0f),inv=1/max(scale,1.0f);
 float cx=scale*(float(ox)+0.5f),cy=scale*(float(oy)+0.5f);
 long x0=max(long(cx-support+0.5f),0L),x1=min(long(cx+support+0.5f),src);
 long y0=max(long(cy-support+0.5f),0L),y1=min(long(cy+support+0.5f),src);
 float sx=0,sy=0;
 for(long ix=x0;ix<x1;++ix)sx+=cubic((float(ix)-cx+0.5f)*inv);
 for(long iy=y0;iy<y1;++iy)sy+=cubic((float(iy)-cy+0.5f)*inv);
 float result=0;
 for(long iy=y0;iy<y1;++iy){float line=0;for(long ix=x0;ix<x1;++ix)line+=x[(iy*src+ix)*c+ch]*(cubic((float(ix)-cx+0.5f)*inv)/sx);
 result+=line*(cubic((float(iy)-cy+0.5f)*inv)/sy);}
 y[id]=result;
}
kernel void relative_table(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],
 constant Params& p [[buffer(8)]],uint id [[thread_position_in_grid]]) {
 long in=p.i[0],span=p.i[1],d=p.i[2];if(id>=span*span*d)return;
 long q=id/(span*d),k=(id/d)%span,e=id%d;
 float src=max(float(in)/float(2*span-1)*(float(q-k+span-1)+0.5f)-0.5f,0.0f);
 long lo=long(src),hi=min(lo+1,in-1);float a=src-float(lo);
 y[id]=(1-a)*x[lo*d+e]+a*x[hi*d+e];
}
