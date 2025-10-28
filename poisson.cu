// poisson_jacobi_autotune.cu
// EXHAUSTIVE COMBOS FROM LISTS + PRINT LISTS + REJECT/ACCEPT DUMPS
// RUNTIME-SAFE PROBE (configurable) + WARMUPS (configurable) + STABLE AUTOTUNE (median + epsilon + tiebreak)
// Build: nvcc -O3 -arch=sm_80 -DTYPE=double poisson_jacobi_autotune.cu -o poisson

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

#ifndef TYPE
#define TYPE double
#endif
using real = TYPE;

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

// ================= KERNELS =================
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

template <typename T>
__global__ void abs_diff_kernel(int n,const T* __restrict__ a,const T* __restrict__ b,T* __restrict__ out){
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  if (t<n) out[t] = fabs(a[t]-b[t]);
}

// ================= UTIL =================
struct BlockCfg{ dim3 block; std::string name; };

static inline const char* dtype_str(){
  if (sizeof(real)==2) return "fp16";
  if (sizeof(real)==4) return "fp32";
  return "fp64";
}
static inline double flops_per_point(){ return 6.0; }
static inline size_t bytes_per_point(){ return 7*sizeof(real); } // 5*u + f + write

static std::vector<int> parse_list(const char* s){
  std::vector<int> v; if(!s) return v;
  int val=0; bool in=false;
  for(const char* p=s;;++p){
    if(*p>='0' && *p<='9'){ val=val*10+(*p-'0'); in=true; }
    else { if(in){ v.push_back(val); val=0; in=false; } if(!*p) break; }
  }
  return v;
}
static bool starts_with(const char* s,const char* pre){ return std::strncmp(s,pre,std::strlen(pre))==0; }
static bool is_number(const char* s){
  if(!s||!*s) return false;
  char* end=nullptr; std::strtod(s,&end);
  return end && *end=='\0';
}
static std::string vec_to_str(const std::vector<int>& v){
  std::string s;
  for(size_t i=0;i<v.size();++i){ if(i) s += ","; s += std::to_string(v[i]); }
  return s;
}

template <typename T>
static void query_kernel_attrs(const void* func,int& numRegs,int& minGridSize,int& suggestedBlockSize,size_t& staticSmem){
  cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr, func));
  numRegs = attr.numRegs; staticSmem = attr.sharedSizeBytes;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize,&suggestedBlockSize,func,0,0));
}

// ===== benchmark (probe y warmups configurables)
template <typename T>
static float benchmark_and_report(int nx,int ny,int bench_iters,
                                  T* d_u,T* d_unew,T* d_f,T h2,
                                  dim3 block,const char* name,
                                  int warmup_iters,int do_probe){
  dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);

  // Probe opcional
  if (do_probe){
    jacobi_step<T><<<grid, block>>>(nx, ny, d_u, d_unew, d_f, h2);
    cudaError_t probe_err = cudaPeekAtLastError();
    if (probe_err != cudaSuccess) {
      fprintf(stderr, "[SKIP] bx=%u by=%u -> %s\n", block.x, block.y, cudaGetErrorString(probe_err));
      (void)cudaGetLastError();
      return INFINITY;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Warmups configurables (no cronometrados)
  for(int w=0; w<warmup_iters; ++w){
    jacobi_step<T><<<grid,block>>>(nx,ny,d_u,d_unew,d_f,h2);
    std::swap(d_u,d_unew);
  }
  if (warmup_iters>0) CUDA_CHECK(cudaDeviceSynchronize());

  // Cronometrado
  cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));
  for(int it=0; it<bench_iters; ++it){
    jacobi_step<T><<<grid,block>>>(nx,ny,d_u,d_unew,d_f,h2);
    std::swap(d_u,d_unew);
  }
  CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
  CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));

  const double time_s = ms*1e-3;
  const size_t interior = (size_t)(nx-2)*(size_t)(ny-2);
  const double flops  = flops_per_point()*(double)interior*bench_iters;
  const double gflops = (time_s>0)? flops/time_s/1e9 : 0.0;
  const double bytes  = (double)bytes_per_point()*(double)interior*bench_iters;
  const double gbps   = (time_s>0)? bytes/time_s/1e9 : 0.0;

  int numRegs=0,minGrid=0,sugg=0; size_t smem=0;
  query_kernel_attrs<real>((const void*)jacobi_step<real>,numRegs,minGrid,sugg,smem);

  printf("SUMMARY imax=%d jmax=%d dtype=%s flops_per_pt=6 points=padded "
         "bx=%u by=%u time_ms=%.3f iters=%d gflops=%.9f bandwidth_GBps=%.9f "
         "avg_iter_ms=%.6f suggested_block_threads=%d num_regs=%d min_grid_size=%d static_smem_B=%zu\n",
         nx, ny, dtype_str(), block.x, block.y, ms, bench_iters, gflops, gbps,
         ms/bench_iters, sugg, numRegs, minGrid, smem);

  return ms/bench_iters;
}

// ================= ARGS =================
struct Opts {
  int nx=2048, ny=2048, max_iters=8000, bench_iters=200;
  double tol=1e-4;
  int autotune=0;
  std::string bx_list_env, by_list_env;
  int check_every=250;

  // NUEVO: control de entorno y estabilidad
  int warmup_iters=0;   // warmups no medidos
  int probe=1;          // 1=probe, 0=no
  int device=-1;        // -1=default; >=0 fija GPU
  int tune_trials=1;    // repeticiones por candidato (usa mediana)
  double eps_ms=0.01;   // umbral de empate (ms)
};

static Opts parse_args(int argc,char** argv){
  Opts o;
  if(const char* a = std::getenv("AUTOTUNE"))      o.autotune = std::atoi(a);
  if(const char* bx = std::getenv("BX_LIST"))      o.bx_list_env = bx;
  if(const char* by = std::getenv("BY_LIST"))      o.by_list_env = by;
  if(const char* w  = std::getenv("WARMUP_ITERS")) o.warmup_iters = std::max(0, std::atoi(w));
  if(const char* p  = std::getenv("PROBE"))        o.probe        = std::max(0, std::atoi(p));
  if(const char* d  = std::getenv("CUDA_DEVICE"))  o.device       = std::atoi(d);
  if(const char* t  = std::getenv("TUNE_TRIALS"))  o.tune_trials  = std::max(1, std::atoi(t));
  if(const char* e  = std::getenv("EPS_MS"))       o.eps_ms       = std::atof(e);

  std::vector<std::string> pos;
  for(int i=1;i<argc;++i){
    const char* s = argv[i];
    if(starts_with(s,"--bx-list="))         o.bx_list_env = std::string(s+11);
    else if(starts_with(s,"--by-list="))    o.by_list_env = std::string(s+11);
    else if(starts_with(s,"--tune-reps="))  o.bench_iters = std::max(1, std::atoi(s+12));
    else if(starts_with(s,"--check-every="))o.check_every = std::max(1, std::atoi(s+14));
    else if(starts_with(s,"--nx="))         o.nx = std::max(1, std::atoi(s+5));
    else if(starts_with(s,"--ny="))         o.ny = std::max(1, std::atoi(s+5));
    else if(starts_with(s,"--max-iters="))  o.max_iters = std::max(1, std::atoi(s+12));
    else if(starts_with(s,"--tol="))        o.tol = std::atof(s+6);
    else if(starts_with(s,"--bench-iters="))o.bench_iters = std::max(1, std::atoi(s+14));
    else if(starts_with(s,"--autotune="))   o.autotune = std::atoi(s+11);
    else if(starts_with(s,"--warmup-iters=")) o.warmup_iters = std::max(0, std::atoi(s+15));
    else if(starts_with(s,"--probe="))        o.probe        = std::max(0, std::atoi(s+8));
    else if(starts_with(s,"--device="))       o.device       = std::atoi(s+9);
    else if(starts_with(s,"--tune-trials="))  o.tune_trials  = std::max(1, std::atoi(s+14));
    else if(starts_with(s,"--eps-ms="))       o.eps_ms       = std::atof(s+9);
    else if (is_number(s)) pos.emplace_back(s);
  }
  if(pos.size()>0) o.nx = std::max(1, std::atoi(pos[0].c_str()));
  if(pos.size()>1) o.ny = std::max(1, std::atoi(pos[1].c_str()));
  if(pos.size()>2) o.max_iters = std::max(1, std::atoi(pos[2].c_str()));
  if(pos.size()>3) o.tol = std::atof(pos[3].c_str());
  if(pos.size()>4) o.bench_iters = std::max(1, std::atoi(pos[4].c_str()));
  return o;
}

// ================= CANDIDATES =================
static std::vector<BlockCfg> make_candidates_filtered(const std::string& bx_s,
                                                      const std::string& by_s){
  fprintf(stderr, "[INFO] BX_LIST(raw)=\"%s\"\n", bx_s.c_str());
  fprintf(stderr, "[INFO] BY_LIST(raw)=\"%s\"\n", by_s.c_str());

  std::vector<int> bx_list = parse_list(bx_s.c_str());
  std::vector<int> by_list = parse_list(by_s.c_str());

  fprintf(stderr, "[INFO] BX_LIST(parsed) count=%zu : %s\n", bx_list.size(), vec_to_str(bx_list).c_str());
  fprintf(stderr, "[INFO] BY_LIST(parsed) count=%zu : %s\n", by_list.size(), vec_to_str(by_list).c_str());

  if (bx_list.empty() || by_list.empty()) {
    fprintf(stderr, "[ERROR] BX_LIST/BY_LIST vacíos tras parseo. Pásalos por --bx-list / --by-list.\n");
    std::exit(EXIT_FAILURE);
  }

  struct Rej { std::string name, reason; };
  std::vector<BlockCfg> out;
  std::vector<Rej> rejected;
  out.reserve((size_t)bx_list.size() * (size_t)by_list.size());
  rejected.reserve((size_t)bx_list.size() * (size_t)by_list.size());

  long long rej_nonpos=0, rej_low=0, rej_mod=0, rej_high=0;

  for(int bx : bx_list){
    for(int by : by_list){
      const std::string nm = std::to_string(bx)+"x"+std::to_string(by);
      if(bx <= 0 || by <= 0){ ++rej_nonpos; rejected.push_back({nm,"nonpos"}); continue; }

      const int thr = bx * by;
      if(thr < 32)        { ++rej_low;  rejected.push_back({nm,"low(<32)"});    continue; }
      if((thr % 32) != 0) { ++rej_mod;  rejected.push_back({nm,"mod(!%32)"});   continue; }
      if(thr > 1024)      { ++rej_high; rejected.push_back({nm,"high(>1024)"}); continue; }

      out.push_back({ dim3(bx,by,1), nm });
    }
  }

  if(out.empty()){
    fprintf(stderr,"[ERROR] Ninguna combinación (bx,by) válida tras filtros por producto.\n");
    fprintf(stderr,"[REJECTS] total=%zu  reasons: nonpos=%lld low(<32)=%lld mod(!%%32)=%lld high(>1024)=%lld\n",
            rejected.size(), rej_nonpos, rej_low, rej_mod, rej_high);
    for(const auto& r: rejected) fprintf(stderr,"  - %s : %s\n", r.name.c_str(), r.reason.c_str());
    std::exit(EXIT_FAILURE);
  }

  // Orden estable: más threads primero; empate -> bx asc, by asc
  std::sort(out.begin(), out.end(), [](const BlockCfg& a, const BlockCfg& b){
    const int ta = a.block.x*a.block.y, tb = b.block.x*b.block.y;
    if(ta != tb) return ta > tb;
    if(a.block.x != b.block.x) return a.block.x < b.block.x;
    return a.block.y < b.block.y;
  });

  fprintf(stderr,"[INFO] candidates(accepted)=%zu  rejections=%zu  reasons: nonpos=%lld low(<32)=%lld mod(!%%32)=%lld high(>1024)=%lld\n",
          out.size(), rejected.size(), rej_nonpos, rej_low, rej_mod, rej_high);

  fprintf(stderr,"[INFO] accepted:");
  for (size_t i=0;i<out.size();++i) fprintf(stderr," %s", out[i].name.c_str());
  fprintf(stderr,"\n");

  if(!rejected.empty()){
    fprintf(stderr,"[INFO] rejected (with reason):\n");
    for (const auto& r : rejected)
      fprintf(stderr,"  - %s : %s\n", r.name.c_str(), r.reason.c_str());
  }

  return out;
}

// ================= MAIN =================
int main(int argc,char** argv){
  Opts opt = parse_args(argc,argv);

  // Device selection (opcional pero recomendado para estabilidad)
  if (opt.device >= 0) CUDA_CHECK(cudaSetDevice(opt.device));
  int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  fprintf(stderr,"[INFO] Using device %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);

  // Problem setup
  const int nx = opt.nx, ny = opt.ny;
  const int max_iters = opt.max_iters;
  const int bench_iters = opt.bench_iters;
  const real tol = (real)opt.tol;

  const size_t n = (size_t)nx*(size_t)ny;
  const size_t bytes = n*sizeof(real);
  const real Lx=1.0, hx = Lx/(nx-1), h2 = hx*hx;

  std::vector<real> h_u(n,0), h_f(n,0);
  for(int j=0;j<ny;++j) h_u[0+j*nx] = (real)1.0;

  real *d_u,*d_unew,*d_f,*d_tmp;
  CUDA_CHECK(cudaMalloc(&d_u,bytes));
  CUDA_CHECK(cudaMalloc(&d_unew,bytes));
  CUDA_CHECK(cudaMalloc(&d_f,bytes));
  CUDA_CHECK(cudaMalloc(&d_tmp,bytes));
  CUDA_CHECK(cudaMemcpy(d_u,h_u.data(),bytes,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_unew,h_u.data(),bytes,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f,h_f.data(),bytes,cudaMemcpyHostToDevice));

  // Candidates
  auto cands = make_candidates_filtered(opt.bx_list_env, opt.by_list_env);

  // ===== Autotune =====
  dim3 best_block = cands.front().block; std::string best_name = cands.front().name; float best_ms=1e30f;
  if(opt.autotune){
    printf("=== Auto-tuning blockDim (nx=%d ny=%d bench_iters=%d, warmups=%d, probe=%d, trials=%d, eps_ms=%.3f) ===\n",
           nx, ny, bench_iters, opt.warmup_iters, opt.probe, opt.tune_trials, opt.eps_ms);
    real *u_b,*un_b; CUDA_CHECK(cudaMalloc(&u_b,bytes)); CUDA_CHECK(cudaMalloc(&un_b,bytes));

    auto square_score = [](dim3 b){ int d = (int)b.x - (int)b.y; return d>=0? d : -d; };

    for(const auto& cfg : cands){
      std::vector<float> trials; trials.reserve((size_t)opt.tune_trials);
      for (int r=0; r<opt.tune_trials; ++r){
        CUDA_CHECK(cudaMemcpy(u_b,d_u,bytes,cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(un_b,d_unew,bytes,cudaMemcpyDeviceToDevice));
        float avg = benchmark_and_report<real>(nx,ny,bench_iters,u_b,un_b,d_f,h2,
                                               cfg.block,cfg.name.c_str(),
                                               opt.warmup_iters, opt.probe);
        if (avg!=INFINITY) trials.push_back(avg);
      }
      if (!trials.empty()){
        std::sort(trials.begin(), trials.end());
        float med = trials[trials.size()/2]; // mediana

        if (med < best_ms - (float)opt.eps_ms){
          best_ms = med; best_block = cfg.block; best_name = cfg.name;
        } else if (std::fabs(med - best_ms) <= (float)opt.eps_ms){
          // Desempate determinista: preferir bloque más “cuadrado”
          if (square_score(cfg.block) < square_score(best_block)){
            best_ms = med; best_block = cfg.block; best_name = cfg.name;
          }
        }
      }
    }
    CUDA_CHECK(cudaFree(u_b)); CUDA_CHECK(cudaFree(un_b));
    if (best_ms == INFINITY || best_ms==1e30f) {
      fprintf(stderr,"[ERROR] Ningún candidato se pudo lanzar.\n");
      std::exit(EXIT_FAILURE);
    }

    // ==== Reporte GLOBAL_BEST con las mismas métricas que SUMMARY ====
    const size_t interior_pts = (size_t)(nx-2) * (size_t)(ny-2);
    const double time_ms_total = (double)best_ms * (double)bench_iters; // total ms del benchmark
    const double time_s_total  = time_ms_total * 1e-3;
    const double flops_total   = flops_per_point() * (double)interior_pts * (double)bench_iters;
    const double gflops_best   = (time_s_total>0) ? (flops_total / time_s_total / 1e9) : 0.0;
    const double bytes_total   = (double)bytes_per_point() * (double)interior_pts * (double)bench_iters;
    const double gbps_best     = (time_s_total>0) ? (bytes_total / time_s_total / 1e9) : 0.0;

    int numRegs=0, minGrid=0, suggested=0; size_t smem=0;
    query_kernel_attrs<real>((const void*)jacobi_step<real>, numRegs, minGrid, suggested, smem);

    printf("GLOBAL_BEST imax=%d jmax=%d dtype=%s flops_per_pt=6 points=padded "
           "bx=%u by=%u time_ms=%.3f iters=%d gflops=%.9f bandwidth_GBps=%.9f "
           "avg_iter_ms=%.6f suggested_block_threads=%d num_regs=%d min_grid_size=%d static_smem_B=%zu\n",
           nx, ny, dtype_str(),
           best_block.x, best_block.y,
           time_ms_total, bench_iters, gflops_best, gbps_best,
           best_ms, suggested, numRegs, minGrid, smem);
  } else {
    fprintf(stderr,"[INFO] AUTOTUNE=0, using default %s\n", best_name.c_str());
  }

  // ===== Full solve =====
  dim3 grid((nx+best_block.x-1)/best_block.x, (ny+best_block.y-1)/best_block.y);
  thrust::device_ptr<real> dptr(d_tmp);
  real maxdiff=0; int it=0;

  cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s));

  for(it=0; it<max_iters; ++it){
    jacobi_step<real><<<grid,best_block>>>(nx,ny,d_u,d_unew,d_f,h2);
    int thr=256, grd=(int)((n+thr-1)/thr);
    abs_diff_kernel<real><<<grd,thr>>>((int)n,d_unew,d_u,d_tmp);
    maxdiff = thrust::reduce(dptr,dptr+n,(real)0,thrust::maximum<real>());
    if (it % std::max(1,opt.check_every) == 0)
      printf("iter %d  max|Δu|=%e\n", it, (double)maxdiff);
    if(maxdiff < tol) break;
    std::swap(d_u,d_unew);
  }

  CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
  float total_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&total_ms,s,e));
  const double time_s = total_ms*1e-3;
  const size_t interior = (size_t)(nx-2)*(size_t)(ny-2);
  const double flops = flops_per_point()*(double)interior*it;
  const double gflops = (time_s>0)? flops/time_s/1e9 : 0.0;
  const double bytes_t = (double)bytes_per_point()*(double)interior*it;
  const double gbps = (time_s>0)? bytes_t/time_s/1e9 : 0.0;

  printf("FINAL bx=%u by=%u iters=%d gflops=%.3f BW=%.3f GB/s time_ms=%.2f\n",
         best_block.x, best_block.y, it, gflops, gbps, total_ms);

  CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_unew)); CUDA_CHECK(cudaFree(d_f)); CUDA_CHECK(cudaFree(d_tmp));
  return 0;
}
