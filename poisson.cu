// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <thrust/device_ptr.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <cuda_runtime.h>

// #ifndef TYPE
// #define TYPE double
// #endif
// using real = TYPE;

// #define CUDA_CHECK(call) do { \
//   cudaError_t err = call; \
//   if (err != cudaSuccess) { \
//     fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
//     std::exit(EXIT_FAILURE); \
//   } \
// } while(0)

// #define KERNEL_TAG /* #[kernel] */

// // ================= KERNELS =================
// KERNEL_TAG
// template <typename T>
// __global__ void jacobi_step(int nx,int ny,const T* __restrict__ u,T* __restrict__ unew,
//                             const T* __restrict__ f,T h2){
//   int i = blockIdx.x*blockDim.x + threadIdx.x;
//   int j = blockIdx.y*blockDim.y + threadIdx.y;
//   if (i>=nx || j>=ny) return;
//   int id = i + j*nx;
//   if (i==0 || i==nx-1 || j==0 || j==ny-1){ unew[id]=u[id]; return; }
//   T sum4 = u[id-1] + u[id+1] + u[id-nx] + u[id+nx];
//   T rhs  = f ? f[id] : (T)0;
//   unew[id] = (sum4 - h2*rhs) * (T)0.25;
// }

// KERNEL_TAG
// template <typename T>
// __global__ void abs_diff_kernel(int n,const T* __restrict__ a,const T* __restrict__ b,T* __restrict__ out){
//   int t = blockIdx.x*blockDim.x + threadIdx.x;
//   if (t<n) out[t] = fabs(a[t]-b[t]);
// }

// // ================= UTIL =================
// struct BlockCfg{ dim3 block; std::string name; };

// static inline const char* dtype_str(){
//   if (sizeof(real)==2) return "fp16";
//   if (sizeof(real)==4) return "fp32";
//   return "fp64";
// }
// static inline double flops_per_point(){ return 6.0; }
// static inline size_t bytes_per_point(){ return 7*sizeof(real); } // 5*u + f + write

// static std::vector<int> parse_list(const char* s){
//   std::vector<int> v; if(!s) return v;
//   int val=0; bool in=false;
//   for(const char* p=s;;++p){
//     if(*p>='0' && *p<='9'){ val=val*10+(*p-'0'); in=true; }
//     else { if(in){ v.push_back(val); val=0; in=false; } if(!*p) break; }
//   }
//   return v;
// }
// static bool starts_with(const char* s,const char* pre){ return std::strncmp(s,pre,std::strlen(pre))==0; }
// static bool is_number(const char* s){
//   if(!s||!*s) return false;
//   char* end=nullptr; std::strtod(s,&end);
//   return end && *end=='\0';
// }
// static std::string vec_to_str(const std::vector<int>& v){
//   std::string s;
//   for(size_t i=0;i<v.size();++i){ if(i) s += ","; s += std::to_string(v[i]); }
//   return s;
// }

// template <typename T>
// static void query_kernel_attrs(const void* func,int& numRegs,int& minGridSize,int& suggestedBlockSize,size_t& staticSmem){
//   cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr, func));
//   numRegs = attr.numRegs; staticSmem = attr.sharedSizeBytes;
//   CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize,&suggestedBlockSize,func,0,0));
// }

// // ================= ARGS =================
// struct Opts {
//   int nx=2048, ny=2048, max_iters=8000;
//   double tol=1e-4;
//   std::string bx_list_env, by_list_env;
//   int check_every=250;
//   int device=-1;        // -1=default; >=0 set GPU
//   // online-tuning printing control
//   double eps_pct=0.3;   // tie threshold in % GFLOPS (stability)
// };

// static Opts parse_args(int argc,char** argv){
//   Opts o;
//   if(const char* bx = std::getenv("BX_LIST"))      o.bx_list_env = bx;
//   if(const char* by = std::getenv("BY_LIST"))      o.by_list_env = by;
//   if(const char* d  = std::getenv("CUDA_DEVICE"))  o.device       = std::atoi(d);
//   if(const char* e  = std::getenv("EPS_PCT"))      o.eps_pct      = std::atof(e);

//   std::vector<std::string> pos;
//   for(int i=1;i<argc;++i){
//     const char* s = argv[i];
//     if(starts_with(s,"--bx-list="))         o.bx_list_env = std::string(s+11);
//     else if(starts_with(s,"--by-list="))    o.by_list_env = std::string(s+11);
//     else if(starts_with(s,"--check-every="))o.check_every = std::max(1, std::atoi(s+14));
//     else if(starts_with(s,"--nx="))         o.nx = std::max(1, std::atoi(s+5));
//     else if(starts_with(s,"--ny="))         o.ny = std::max(1, std::atoi(s+5));
//     else if(starts_with(s,"--max-iters="))  o.max_iters = std::max(1, std::atoi(s+12));
//     else if(starts_with(s,"--tol="))        o.tol = std::atof(s+6);
//     else if(starts_with(s,"--device="))     o.device = std::atoi(s+9);
//     else if(starts_with(s,"--eps-pct="))    o.eps_pct = std::atof(s+10);
//     else if (is_number(s)) pos.emplace_back(s);
//   }
//   if(pos.size()>0) o.nx = std::max(1, std::atoi(pos[0].c_str()));
//   if(pos.size()>1) o.ny = std::max(1, std::atoi(pos[1].c_str()));
//   if(pos.size()>2) o.max_iters = std::max(1, std::atoi(pos[2].c_str()));
//   if(pos.size()>3) o.tol = std::atof(pos[3].c_str());
//   return o;
// }

// // ================= CANDIDATES =================
// static std::vector<BlockCfg> make_candidates_filtered(const std::string& bx_s,
//                                                       const std::string& by_s){
//   fprintf(stderr, "[INFO] BX_LIST(raw)=\"%s\"\n", bx_s.c_str());
//   fprintf(stderr, "[INFO] BY_LIST(raw)=\"%s\"\n", by_s.c_str());

//   std::vector<int> bx_list = parse_list(bx_s.c_str());
//   std::vector<int> by_list = parse_list(by_s.c_str());

//   fprintf(stderr, "[INFO] BX_LIST(parsed) count=%zu : %s\n", bx_list.size(), vec_to_str(bx_list).c_str());
//   fprintf(stderr, "[INFO] BY_LIST(parsed) count=%zu : %s\n", by_list.size(), vec_to_str(by_list).c_str());

//   if (bx_list.empty() || by_list.empty()) {
//     fprintf(stderr, "[ERROR] BX_LIST/BY_LIST are empty. Provide --bx-list / --by-list.\n");
//     std::exit(EXIT_FAILURE);
//   }

//   struct Rej { std::string name, reason; };
//   std::vector<BlockCfg> out;
//   std::vector<Rej> rejected;
//   out.reserve((size_t)bx_list.size() * (size_t)by_list.size());
//   rejected.reserve((size_t)bx_list.size() * (size_t)by_list.size());

//   long long rej_nonpos=0, rej_low=0, rej_mod=0, rej_high=0;

//   for(int bx : bx_list){
//     for(int by : by_list){
//       const std::string nm = std::to_string(bx)+"x"+std::to_string(by);
//       if(bx <= 0 || by <= 0){ ++rej_nonpos; rejected.push_back({nm,"nonpos"}); continue; }
//       const int thr = bx * by;
//       if(thr < 32)        { ++rej_low;  rejected.push_back({nm,"low(<32)"});    continue; }
//       if((thr % 32) != 0) { ++rej_mod;  rejected.push_back({nm,"mod(!%32)"});   continue; }
//       if(thr > 1024)      { ++rej_high; rejected.push_back({nm,"high(>1024)"}); continue; }
//       out.push_back({ dim3(bx,by,1), nm });
//     }
//   }

//   if(out.empty()){
//     fprintf(stderr,"[ERROR] No valid (bx,by) after filtering.\n");
//     fprintf(stderr,"[REJECTS] total=%zu  reasons: nonpos=%lld low(<32)=%lld mod(!%%32)=%lld high(>1024)=%lld\n",
//             rejected.size(), rej_nonpos, rej_low, rej_mod, rej_high);
//     for(const auto& r: rejected) fprintf(stderr,"  - %s : %s\n", r.name.c_str(), r.reason.c_str());
//     std::exit(EXIT_FAILURE);
//   }

//   // Stable order: more threads first; tie -> bx asc, by asc
//   std::sort(out.begin(), out.end(), [](const BlockCfg& a, const BlockCfg& b){
//     const int ta = a.block.x*a.block.y, tb = b.block.x*b.block.y;
//     if(ta != tb) return ta > tb;
//     if(a.block.x != b.block.x) return a.block.x < b.block.x;
//     return a.block.y < b.block.y;
//   });

//   fprintf(stderr,"[INFO] candidates(accepted)=%zu  rejections=%zu  reasons: nonpos=%lld low(<32)=%lld mod(!%%32)=%lld high(>1024)=%lld\n",
//           out.size(), rejected.size(), rej_nonpos, rej_low, rej_mod, rej_high);

//   fprintf(stderr,"[INFO] accepted:");
//   for (size_t i=0;i<out.size();++i) fprintf(stderr," %s", out[i].name.c_str());
//   fprintf(stderr,"\n");

//   if(!rejected.empty()){
//     fprintf(stderr,"[INFO] rejected (with reason):\n");
//     for (const auto& r : rejected)
//       fprintf(stderr,"  - %s : %s\n", r.name.c_str(), r.reason.c_str());
//   }

//   return out;
// }

// // ================= ONLINE TUNER (single kernel launch per iteration) =================
// struct Acc {
//   double time_ms=0.0, gflops=0.0, bw=0.0;
// };

// struct OnlineTuner {
//   std::vector<BlockCfg> cand;
//   std::vector<Acc> acc;
//   int cur=0, best=0;
//   bool tuned=false;
//   dim3 block{256,4,1};
//   std::string name="256x4";
// };

// static inline double grid_waste(int nx,int ny, dim3 b){
//   double gx = std::ceil(nx/(double)b.x);
//   double gy = std::ceil(ny/(double)b.y);
//   double launched = gx*gy*b.x*b.y;
//   double useful = (double)nx*(double)ny;
//   return launched - useful; // lower is better
// }

// // returns the time (ms) of this iteration's single jacobi launch
// template <typename T>
// static float tuner_step_once(OnlineTuner& t,
//                              int nx,int ny,
//                              T*& d_u,T*& d_unew,const T* d_f,T h2)
// {
//   // choose block for THIS iteration
//   if(!t.tuned){
//     t.block = t.cand[t.cur].block;
//     t.name  = t.cand[t.cur].name;
//   }
//   dim3 grid((nx+t.block.x-1)/t.block.x, (ny+t.block.y-1)/t.block.y);

//   // time a single jacobi_step launch (and swap)
//   cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
//   CUDA_CHECK(cudaEventRecord(s));
//   jacobi_step<T><<<grid,t.block>>>(nx,ny,d_u,d_unew,d_f,h2);
//   std::swap(d_u,d_unew);
//   CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
//   float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
//   CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));

//   // if still tuning, record metrics and maybe finalize
//   if(!t.tuned){
//     const size_t interior = (size_t)(nx-2)*(size_t)(ny-2);
//     const double sec = ms*1e-3;
//     const double flops = flops_per_point()*(double)interior; // per-iter
//     const double gflops = (sec>0)? flops/sec/1e9 : 0.0;
//     const double bytes  = (double)bytes_per_point()*(double)interior;
//     const double gbps   = (sec>0)? bytes/sec/1e9 : 0.0;
//     t.acc[t.cur] = { (double)ms, gflops, gbps };

//     printf("TUNE cand=%d  bx=%u by=%u  time_ms=%.3f  gflops=%.3f  BW=%.3f GB/s\n",
//            t.cur, t.block.x, t.block.y, ms, gflops, gbps);

//     // next candidate or finish
//     ++t.cur;
//     if(t.cur >= (int)t.cand.size()){
//       // pick best: max gflops, tie -> min time, then min grid waste, then prefer larger bx
//       t.best = 0;
//       for(int i=1;i<(int)t.cand.size();++i){
//         const bool better = (t.acc[i].gflops > t.acc[t.best].gflops);
//         const bool tieGF  = fabs(t.acc[i].gflops - t.acc[t.best].gflops) <=
//                             (t.acc[t.best].gflops * 0.003); // ~0.3% tie
//         if (better) { t.best = i; continue; }
//         if (tieGF){
//           if (t.acc[i].time_ms + 1e-6 < t.acc[t.best].time_ms) { t.best = i; continue; }
//           double wi = grid_waste(nx,ny,t.cand[i].block);
//           double wb = grid_waste(nx,ny,t.cand[t.best].block);
//           if (wi + 0.5 < wb) { t.best = i; continue; }
//           if (t.cand[i].block.x > t.cand[t.best].block.x) { t.best = i; continue; }
//         }
//       }
//       t.block = t.cand[t.best].block;
//       t.name  = t.cand[t.best].name;
//       t.tuned = true;

//       printf("\nGLOBAL_BEST (online) bx=%u by=%u  gflops=%.3f  BW=%.3f GB/s  time_ms=%.3f\n\n",
//              t.block.x, t.block.y, t.acc[t.best].gflops, t.acc[t.best].bw, t.acc[t.best].time_ms);

//       // (optional) print kernel attrs
//       int numRegs=0, minGrid=0, suggested=0; size_t smem=0;
//       query_kernel_attrs<real>((const void*)jacobi_step<real>, numRegs, minGrid, suggested, smem);
//       printf("KATTR num_regs=%d static_smem_B=%zu suggested_block_threads=%d min_grid_size=%d\n",
//              numRegs, smem, suggested, minGrid);
//     }
//   }

//   return ms;
// }



// // ================= MAIN =================
// int main(int argc,char** argv){
//   Opts opt = parse_args(argc,argv);

//   // Device selection (optional but recommended)
//   if (opt.device >= 0) CUDA_CHECK(cudaSetDevice(opt.device));
//   int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
//   cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
//   fprintf(stderr,"[INFO] Using device %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);

//   // Problem setup
//   const int nx = opt.nx, ny = opt.ny;
//   const int max_iters = opt.max_iters;
//   const real tol = (real)opt.tol;

//   const size_t n = (size_t)nx*(size_t)ny;
//   const size_t bytes = n*sizeof(real);
//   const real Lx=1.0, hx = Lx/(nx-1), h2 = hx*hx;

//   std::vector<real> h_u(n,0), h_f(n,0);
//   for(int j=0;j<ny;++j) h_u[0+j*nx] = (real)1.0;

//   real *d_u,*d_unew,*d_f,*d_tmp;
//   CUDA_CHECK(cudaMalloc(&d_u,bytes));
//   CUDA_CHECK(cudaMalloc(&d_unew,bytes));
//   CUDA_CHECK(cudaMalloc(&d_f,bytes));
//   CUDA_CHECK(cudaMalloc(&d_tmp,bytes));
//   CUDA_CHECK(cudaMemcpy(d_u,h_u.data(),bytes,cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaMemcpy(d_unew,h_u.data(),bytes,cudaMemcpyHostToDevice));
//   CUDA_CHECK(cudaMemcpy(d_f,h_f.data(),bytes,cudaMemcpyHostToDevice));

//   // Candidates (must be provided)
//   auto cands = make_candidates_filtered(opt.bx_list_env, opt.by_list_env);

//   // Online tuner state
//   OnlineTuner tuner;
//   tuner.cand = cands;
//   tuner.acc.resize(cands.size());
//   tuner.cur = 0;
//   tuner.best = 0;
//   tuner.tuned = (cands.size() <= 1);
//   tuner.block = cands.front().block;
//   tuner.name  = cands.front().name;

//   thrust::device_ptr<real> dptr(d_tmp);
//   real maxdiff = (real)1e30;
//   int it = 0;
//   const int thr = 256;
//   const int grd = (int)((n + thr - 1)/thr);

//   cudaEvent_t es,ee; CUDA_CHECK(cudaEventCreate(&es)); CUDA_CHECK(cudaEventCreate(&ee));
//   CUDA_CHECK(cudaEventRecord(es));

//   // ===== Main loop (simple): one jacobi_step per iteration; error checked every check_every =====
//   while (it < max_iters) {
//     // one call that: sets block (if tuning), runs the kernel once, times it, swaps u/unew, and possibly finalizes best
//     (void)tuner_step_once<real>(tuner, nx, ny, d_u, d_unew, d_f, h2);

//     // error check cadence (same idea as original)
//     if (it % std::max(1,opt.check_every) == 0) {
//       abs_diff_kernel<real><<<grd,thr>>>((int)n,d_unew,d_u,d_tmp);
//       maxdiff = thrust::reduce(dptr,dptr+n,(real)0,thrust::maximum<real>());
//       printf("iter %d  max|Δu|=%e  %s\n", it, (double)maxdiff,
//              tuner.tuned ? "[locked]" : ("[tuning " + tuner.name + "]").c_str());
//       if (maxdiff < tol) break;
//     }

//     ++it;
//   }

//   CUDA_CHECK(cudaEventRecord(ee)); CUDA_CHECK(cudaEventSynchronize(ee));
//   float total_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&total_ms,es,ee));

//   // Final metrics (include the online tuning iterations)
//   const double time_s = total_ms*1e-3;
//   const size_t interior = (size_t)(nx-2)*(size_t)(ny-2);
//   const double flops = flops_per_point()*(double)interior*it;
//   const double gflops = (time_s>0)? flops/time_s/1e9 : 0.0;
//   const double bytes_t = (double)bytes_per_point()*(double)interior*it;
//   const double gbps = (time_s>0)? bytes_t/time_s/1e9 : 0.0;

//   printf("FINAL bx=%u by=%u iters=%d gflops=%.3f BW=%.3f GB/s time_ms=%.2f\n",
//          tuner.block.x, tuner.block.y, it, gflops, gbps, total_ms);

//   CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_unew)); CUDA_CHECK(cudaFree(d_f)); CUDA_CHECK(cudaFree(d_tmp));
//   return 0;
// }


// =====================================================================
// poisson.cu — stable per-kernel TUNE + GLOBAL_BEST logs
//              (batched timing to avoid microsecond artifacts)
//              REAL_T plumbed from compile flags, safe for half/float/double
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

#ifdef HAS_HALF
  #include <cuda_fp16.h>
#endif

// ---- scalar type plumbed from compile defs ----
// Prefer REAL_T (pass -DREAL_T=double|float|__half), fallback to TYPE, else double.
#ifndef REAL_T
  #ifdef TYPE
    #define REAL_T TYPE
  #else
    #define REAL_T double
  #endif
#endif

using real = REAL_T;

#define CUDA_CHECK(call) do { \
  cudaError_t _cuda_err = (call); \
  if (_cuda_err != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_cuda_err), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

#define KERNEL_TAG /* #[kernel] */

// ================= KERNELS =================
KERNEL_TAG
template <typename T>
__global__ void jacobi_step(int nx,int ny,const T* __restrict__ u,T* __restrict__ unew,
                            const T* __restrict__ f,T h2){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i>=nx || j>=ny) return;
  int id = i + j*nx;
  if (i==0 || i==nx-1 || j==0 || j==ny-1){ unew[id]=u[id]; return; }
  T sum4 = u[id-1] + u[id+1] + u[id-nx] + u[id+nx];
  T rhs  = f ? f[id] : (T)0;
  unew[id] = (sum4 - h2*rhs) * (T)0.25;
}

// device abs that works for float/double/half
template <typename T>
__device__ inline T my_abs(T x) { return fabs(x); }   // OK for float/double via promotion

#ifdef HAS_HALF
template <>
__device__ inline __half my_abs<__half>(__half x) { return __habs(x); }
#endif

KERNEL_TAG
template <typename T>
__global__ void abs_diff_kernel(int n,const T* __restrict__ a,const T* __restrict__ b,T* __restrict__ out){
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t<n) out[t] = my_abs<T>(a[t]-b[t]);
}

// ================= HELPERS =================
static inline const char* dtype_str(){
#if defined(HAS_HALF)
  if (std::is_same<real,__half>::value) return "half";
#endif
  if (sizeof(real)==2) return "half";
  if (sizeof(real)==4) return "float";
  return "double";
}
static inline double flops_per_point(){ return 6.0; }                 // ~5-pt Jacobi
static inline size_t bytes_per_point(){ return 7*sizeof(real); }      // 5 reads + f + 1 write

static std::vector<int> parse_list(const char* s){
  std::vector<int> v; if(!s) return v;
  int val=0; bool in=false;
  for(const char* p=s;;++p){
    if(*p>='0' && *p<='9'){ val=val*10+(*p-'0'); in=true; }
    else { if(in){ v.push_back(val); val=0; in=false; } if(!*p) break; }
  }
  return v;
}

struct BlockCfg{ dim3 block; std::string name; };

static std::vector<BlockCfg> make_candidates_filtered(const std::string& bx_s,
                                                      const std::string& by_s){
  fprintf(stderr, "[INFO] BX_LIST(raw)=\"%s\"\n", bx_s.c_str());
  fprintf(stderr, "[INFO] BY_LIST(raw)=\"%s\"\n", by_s.c_str());

  std::vector<int> bx_list = parse_list(bx_s.c_str());
  std::vector<int> by_list = parse_list(by_s.c_str());

  auto to_str=[&](const std::vector<int>& v){ std::string s; for(size_t i=0;i<v.size();++i){ if(i) s+=','; s+=std::to_string(v[i]); } return s; };
  fprintf(stderr, "[INFO] BX_LIST(parsed) count=%zu : %s\n", bx_list.size(), to_str(bx_list).c_str());
  fprintf(stderr, "[INFO] BY_LIST(parsed) count=%zu : %s\n", by_list.size(), to_str(by_list).c_str());

  struct Rej{ std::string name, reason; };
  std::vector<BlockCfg> out; out.reserve((size_t)bx_list.size()*by_list.size());
  std::vector<Rej> rejected; rejected.reserve((size_t)bx_list.size()*by_list.size());

  long long rej_nonpos=0, rej_low=0, rej_mod=0, rej_high=0;

  for(int bx: bx_list){
    for(int by: by_list){
      const std::string nm = std::to_string(bx)+"x"+std::to_string(by);
      if (bx<=0 || by<=0) { ++rej_nonpos; rejected.push_back({nm,"nonpos"}); continue; }
      const int thr = bx*by;
      if (thr < 32)        { ++rej_low;  rejected.push_back({nm,"low(<32)"});    continue; }
      if ((thr % 32) != 0) { ++rej_mod;  rejected.push_back({nm,"mod(!%32)"});   continue; }
      if (thr > 1024)      { ++rej_high; rejected.push_back({nm,"high(>1024)"}); continue; }
      out.push_back({ dim3(bx,by,1), nm });
    }
  }

  fprintf(stderr,"[INFO] candidates(accepted)=%zu  rejections=%zu  reasons: nonpos=%lld low(<32)=%lld mod(!%%32)=%lld high(>1024)=%lld\n",
          out.size(), rejected.size(), rej_nonpos, rej_low, rej_mod, rej_high);

  if(!out.empty()){
    fprintf(stderr,"[INFO] accepted:");
    for (auto& c: out) fprintf(stderr," %s", c.name.c_str());
    fprintf(stderr,"\n");
  }
  if(!rejected.empty()){
    fprintf(stderr,"[INFO] rejected (with reason):\n");
    for (auto& r: rejected) fprintf(stderr,"  - %s : %s\n", r.name.c_str(), r.reason.c_str());
  }
  if(out.empty()){
    fprintf(stderr,"[ERROR] No valid (bx,by) after filtering.\n");
    std::exit(EXIT_FAILURE);
  }
  // stable order: more threads first; tie -> bx asc, by asc
  std::sort(out.begin(), out.end(), [](const BlockCfg& a,const BlockCfg& b){
    const int ta=a.block.x*a.block.y, tb=b.block.x*b.block.y;
    if (ta!=tb) return ta>tb;
    if (a.block.x!=b.block.x) return a.block.x<b.block.x;
    return a.block.y<b.block.y;
  });
  return out;
}

// -------- make_real: host-safe constructor for 'real' from double --------
template <typename R>
inline R make_real(double x) { return (R)x; }

#ifdef HAS_HALF
template <>
inline __half make_real<__half>(double x) { return __float2half((float)x); }
#endif

// -------- Robust timer --------
// Batches launches to ensure long enough window; returns average per launch (ms).
static double time_kernel_ms(std::function<void()> launch,
                             int warmups = 6,
                             int inner = 10,
                             double target_total_ms = 200.0) {
  // Warm-up to stabilize clocks/caches
  for (int i = 0; i < warmups; ++i) { launch(); }
  CUDA_CHECK(cudaDeviceSynchronize());

  int reps = 1;
  double avg_ms = 0.0;

  while (true) {
    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    CUDA_CHECK(cudaEventRecord(startEvt));
    for (int r = 0; r < reps; ++r) {
      for (int k = 0; k < inner; ++k) { launch(); }
    }
    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaEventSynchronize(stopEvt));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, startEvt, stopEvt));
    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));

    if (total_ms >= target_total_ms || reps >= (1 << 18)) {
      avg_ms = (double)total_ms / (double)(reps * inner);
      break;
    }
    reps <<= 1;
  }
  return avg_ms;
}

// ================= PER-KERNEL TUNERS =================
static inline double flops_jacobi(size_t interior){
  return flops_per_point() * (double)interior;
}
static inline double bytes_jacobi(size_t interior){
  return (double)bytes_per_point() * (double)interior;
}

template <typename T>
static void tune_jacobi_step(int nx,int ny,
                             T* d_u, T* d_unew, const T* d_f, T h2,
                             const std::vector<BlockCfg>& cand)
{
  const size_t interior = (size_t)(nx-2)*(size_t)(ny-2);
  const double FLOPS = flops_jacobi(interior);
  const double BYTES = bytes_jacobi(interior);

  int best=-1; double best_gflops=-1.0, best_ms=1e30, best_bw=0.0;
  for(size_t i=0;i<cand.size();++i){
    dim3 blk = cand[i].block;
    dim3 grd( (nx+blk.x-1)/blk.x, (ny+blk.y-1)/blk.y, 1);

    // Average per-launch time over a large, stable window
    double ms = time_kernel_ms([&](){
      jacobi_step<T><<<grd,blk>>>(nx,ny,d_u,d_unew,d_f,h2);
      // swap so subsequent launches continue advancing the state
      T* tmp=d_u; d_u=d_unew; d_unew=tmp;
    });

    const double sec = ms*1e-3;
    const double gflops = (sec>0)? FLOPS/sec/1e9 : 0.0;
    const double bw_gbps= (sec>0)? BYTES/sec/1e9 : 0.0;

    printf("TUNE cand=%zu  bx=%u by=%u  time_ms=%.3f  gflops=%.3f  BW=%.3f GB/s\n",
           i, blk.x, blk.y, ms, gflops, bw_gbps);

    if (gflops > best_gflops || (fabs(gflops-best_gflops)<=best_gflops*0.003 && ms<best_ms)){
      best=(int)i; best_gflops=gflops; best_ms=ms; best_bw=bw_gbps;
    }
  }
  if (best>=0){
    printf("\nGLOBAL_BEST (online) bx=%u by=%u  gflops=%.3f  BW=%.3f GB/s  time_ms=%.3f\n\n",
           cand[best].block.x, cand[best].block.y, best_gflops, best_bw, best_ms);
  }
}

template <typename T>
static void tune_abs_diff(int n,
                          const T* a, const T* b, T* out,
                          const std::vector<BlockCfg>& cand)
{
  // For 1D kernel, use threads_per_block = bx*by
  const double BYTES_PER_ELEM = 3.0 * sizeof(T); // 2 loads + 1 store

  int best=-1; double best_bw=-1.0, best_ms=1e30;
  for(size_t i=0;i<cand.size();++i){
    const int tpb = cand[i].block.x * cand[i].block.y;  // <= 1024 ensured by filter
    dim3 blk(tpb,1,1);
    dim3 grd( (n + tpb - 1)/tpb, 1, 1 );

    double ms = time_kernel_ms([&](){
      abs_diff_kernel<T><<<grd,blk>>>(n,a,b,out);
    });

    const double sec = ms*1e-3;
    const double bw_gbps = (sec>0)? (BYTES_PER_ELEM*(double)n)/sec/1e9 : 0.0;

    // keep gflops=0.0 for this kernel; focus on BW
    printf("TUNE cand=%zu  bx=%u by=%u  time_ms=%.3f  gflops=%.3f  BW=%.3f GB/s\n",
           i, cand[i].block.x, cand[i].block.y, ms, 0.0, bw_gbps);

    if (bw_gbps > best_bw || (fabs(bw_gbps-best_bw)<=best_bw*0.003 && ms<best_ms)){
      best=(int)i; best_bw=bw_gbps; best_ms=ms;
    }
  }
  if (best>=0){
    printf("\nGLOBAL_BEST (online) bx=%u by=%u  gflops=%.3f  BW=%.3f GB/s  time_ms=%.3f\n\n",
           cand[best].block.x, cand[best].block.y, 0.0, best_bw, best_ms);
  }
}

// ================= MAIN =================
int main(int, char**){
  // Problem
  const int nx = 2048, ny = 2048;
  const size_t n = (size_t)nx * (size_t)ny;

  // Do geometry in double on host; cast once to 'real' at the end
  const double Lx_d = 1.0;
  const double hx_d = Lx_d / double(nx - 1);
  const real   h2   = make_real<real>(hx_d * hx_d);

  // Host init (non-trivial boundary)
  std::vector<real> h_u(n, make_real<real>(0.0));
  std::vector<real> h_f(n, make_real<real>(0.0));
  for(int j=0;j<ny;++j) h_u[0 + j*nx] = make_real<real>(1.0);

  // Device buffers
  real *d_u,*d_unew,*d_f,*d_tmp;
  CUDA_CHECK(cudaMalloc(&d_u,   n*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&d_unew,n*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&d_f,   n*sizeof(real)));
  CUDA_CHECK(cudaMalloc(&d_tmp, n*sizeof(real)));

  CUDA_CHECK(cudaMemcpy(d_u,   h_u.data(), n*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_unew,h_u.data(), n*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f,   h_f.data(), n*sizeof(real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_tmp, 0,          n*sizeof(real)));

  // Device info
  int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
  fprintf(stderr,"[INFO] Using device %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);

  // Candidate lists from env (required)
  const char* bx_env = std::getenv("BX_LIST");
  const char* by_env = std::getenv("BY_LIST");
  if (!bx_env || !by_env){
    fprintf(stderr,"[ERROR] Provide BX_LIST and BY_LIST env vars, e.g.\n");
    fprintf(stderr,"        BX_LIST=\"1,2,4,8,16,32,64,128,256,512,1024\"\n");
    fprintf(stderr,"        BY_LIST=\"1,2,4,8,16,32,64,128,256,512,1024\"\n");
    return 1;
  }
  auto candidates = make_candidates_filtered(bx_env, by_env);

  // One sync to settle before timing
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Kernel 1: jacobi_step ----
  printf("=== KERNEL: jacobi_step (%s) ===\n", dtype_str());
  tune_jacobi_step<real>(nx, ny, d_u, d_unew, d_f, (real)h2, candidates);

  // ---- Kernel 2: abs_diff_kernel ----
  printf("=== KERNEL: abs_diff_kernel (%s) ===\n", dtype_str());
  tune_abs_diff<real>((int)n, d_unew, d_u, d_tmp, candidates);

  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_unew));
  CUDA_CHECK(cudaFree(d_f));
  CUDA_CHECK(cudaFree(d_tmp));
  return 0;
}
