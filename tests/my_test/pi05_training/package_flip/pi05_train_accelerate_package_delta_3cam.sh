#!/bin/bash
# ============================================================================
# 翻包项目 PI05 Delta Action + 3 Cameras 多卡训练脚本（rigid_LW_plane）
# ============================================================================
# 数据集: /mnt/data3/jiangxirui/rigid_LW_plane（134 episodes / 91,536 帧）
# 默认卡: GPU 0,1,2（3 卡 DDP × batch_size=16 × grad_accum=2 → 有效 batch=96）
#
# 用法:
#   bash pi05_train_accelerate_package_delta_3cam.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash pi05_train_accelerate_package_delta_3cam.sh
#   bash pi05_train_accelerate_package_delta_3cam.sh --steps=10000     # 透传给训练
# ============================================================================

set -e

# ----------------------------------------------------------------------------
# 关键: 强制使用本仓库(fork)的 src/，而不是 conda 环境里 PyPI 安装的 lerobot
# （fork 的 action_error_metrics / rtc / relative_action 等定制 PyPI 版没有）
# ----------------------------------------------------------------------------
export LEROBOT_REPO="${LEROBOT_REPO:-/home/jiangxirui/g1/lerobot_v0.5.2_fork}"
export PYTHONPATH="${LEROBOT_REPO}/src:${PYTHONPATH:-}"

# conda 环境（含 torch/accelerate 等依赖；本脚本不依赖 conda activate，
# 直接用解释器的绝对路径，避免 PATH 里混入 .local/bin 旧版）
LEROBOT_ENV="${LEROBOT_ENV:-$HOME/miniconda3/envs/lerobot}"
ACCELERATE="${ACCELERATE:-$LEROBOT_ENV/bin/accelerate}"
[ -x "$ACCELERATE" ] || { echo "accelerate 不存在: $ACCELERATE（检查 LEROBOT_ENV）"; exit 1; }

echo "Using accelerate: $ACCELERATE"
echo "Using Python:     $LEROBOT_ENV/bin/python"
"$LEROBOT_ENV/bin/python" -c "import lerobot; print('lerobot from:', lerobot.__file__)" \
    | grep -q "^lerobot from: ${LEROBOT_REPO}/src" \
    || { echo "!! lerobot 未从本仓库 src/ 加载，中止（检查 PYTHONPATH）"; exit 1; }

# ----------------------------------------------------------------------------
# 环境变量
# ----------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# SwanLab key（显式指定，沿用团队脚本既有约定；外部环境变量可覆盖）
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-AYD2zteWSNmfQwa12UgFM}"

# GPU: 默认使用 0,1,2 三张卡 → 进程数 3，有效 batch = 16 × 3 × 2 = 96
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"   # 有效batch=16×3×2=96

# ----------------------------------------------------------------------------
# 数据就绪检查
# ----------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/mnt/data3/jiangxirui/rigid_LW_plane}"
[ -f "$DATA_ROOT/meta/info.json" ] || {
    echo "数据未就绪: $DATA_ROOT 缺少 meta/info.json"
    exit 1
}

# ----------------------------------------------------------------------------
# 启动（输出目录带时间戳，落在 /mnt/data3/jiangxirui，避免混写与撑爆系统盘）
# ----------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data3/jiangxirui/train_outputs}"
mkdir -p "$OUTPUT_ROOT"
RUN_NAME="$(date +%Y%m%d_%H%M%S)_${NUM_PROCESSES}gpu_rigid_lw_plane_delta3cam"

"$ACCELERATE" launch \
    --num_processes="${NUM_PROCESSES}" \
    --multi_gpu \
    --mixed_precision="${MIXED_PRECISION}" \
    --num_machines=1 \
    --machine_rank=0 \
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    -m lerobot.scripts.lerobot_train \
    --config "$(dirname "$0")/package_flip_delta_3cam_config.yaml" \
    --dataset.root "${DATA_ROOT}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --output_dir "${OUTPUT_ROOT}/${RUN_NAME}" \
    "$@"
