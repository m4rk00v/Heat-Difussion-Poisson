#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=poisson_job
#SBATCH --output=poisson_out_gpu_cu.txt
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1   # 1 GPU para estabilidad

echo "Node: $SLURM_NODELIST"
nvidia-smi || true

# === Toolchain ===
module load nvhpc || module load cuda || true

# === Numeric type ===
TYPE="${TYPE:-double}"   # float | double | half
REAL_T_DEF="double"
NVCC_EXTRA=""
case "$TYPE" in
  float)  REAL_T_DEF="float" ;;
  double) REAL_T_DEF="double" ;;
  half)   REAL_T_DEF="__half"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
esac

DTYPE_STR="fp32"
[[ "$TYPE" == "double" ]] && DTYPE_STR="fp64"
[[ "$TYPE" == "half"   ]] && DTYPE_STR="fp16"

# === Compile ===
SRC=${SRC:-poisson.cu}
BIN=${BIN:-poisson}
ARCH=${ARCH:-sm_80}
echo "Compiling $SRC -> $BIN (TYPE=$TYPE, ARCH=$ARCH)"
nvcc -O3 -arch=${ARCH} -Xptxas=-v $NVCC_EXTRA \
  -DREAL_T=$REAL_T_DEF -DDTYPE_STR="\"$DTYPE_STR\"" \
  "$SRC" -o "$BIN" || exit 2

# === Candidate block sizes (listas completas) ===
BX_LIST="${BX_LIST:-,1,2,4,8,16,32,64,128,256,512,1024}"
BY_LIST="${BY_LIST:-,1,2,4,8,16,32,64,128,256,512,1024}"

# === Autotune & solver control ===
TUNE_REPS=${TUNE_REPS:-2}        # iteraciones medidas por candidato (rápido)
WARMUP_ITERS=${WARMUP_ITERS:-0}  # warmups no medidos
PROBE=${PROBE:-1}                # 1=verifica config antes de medir
TUNE_TRIALS=${TUNE_TRIALS:-3}    # repeticiones por candidato (se toma mediana)
EPS_MS=${EPS_MS:-0.02}           # umbral empate (ms)
NX=${NX:-2048}
NY=${NY:-2048}
MAX_ITERS=${MAX_ITERS:-8000}
TOL=${TOL:-1e-4}
AUTOTUNE=${AUTOTUNE:-1}

# === Estabilidad de dispositivo ===
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # fija GPU lógica 0
CUDA_DEVICE=${CUDA_DEVICE:-0}                           # index que verá el binario

echo "GPU logical mask: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
env | grep -E 'BX_LIST|BY_LIST|TUNE_REPS|WARMUP_ITERS|PROBE|TUNE_TRIALS|EPS_MS|CUDA_VISIBLE_DEVICES|CUDA_DEVICE|AUTOTUNE' || true

# === Run ===
echo "Running $BIN with NX=$NX NY=$NY type=$TYPE autotune reps=$TUNE_REPS trials=$TUNE_TRIALS warmups=$WARMUP_ITERS probe=$PROBE eps=$EPS_MS"
export BX_LIST BY_LIST TUNE_REPS WARMUP_ITERS PROBE TUNE_TRIALS EPS_MS AUTOTUNE CUDA_DEVICE

srun ./"$BIN" \
     --bx-list="$BX_LIST" \
     --by-list="$BY_LIST" \
     --tune-reps=$TUNE_REPS \
     --warmup-iters=$WARMUP_ITERS \
     --probe=$PROBE \
     --tune-trials=$TUNE_TRIALS \
     --eps-ms=$EPS_MS \
     --device=$CUDA_DEVICE \
     --check-every=5 \
     $NX $NY $MAX_ITERS $TOL
