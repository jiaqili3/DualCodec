#!/usr/bin/env bash
# Launch DualCodec-SenseVoice training on 4 GPUs.
set -euo pipefail

ROOT="/F00120260003/flexislm_project/jiaqi/DualCodec"
PYTHON="/F00120260003/flexislm_project/miniconda3/envs/fslm/bin/python"
ACCELERATE="/F00120260003/flexislm_project/miniconda3/envs/fslm/bin/accelerate"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# Avoid torchcodec mp3 path; we decode via soundfile/librosa in the dataset.
export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

NUM_PROCESSES="${NUM_PROCESSES:-4}"
EXTRA_ARGS=("$@")

echo "[dualcodec-sensevoice] starting at $(date) on GPUs ${CUDA_VISIBLE_DEVICES}"
echo "[dualcodec-sensevoice] python=${PYTHON}"

"${ACCELERATE}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --multi_gpu \
  --mixed_precision no \
  train.py \
  --config-name=dualcodec_train_sensevoice \
  "${EXTRA_ARGS[@]}"
