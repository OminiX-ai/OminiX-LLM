#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=llm-pretrain
#SBATCH --nodelist=nnmc[90-97]
#SBATCH --partition=all
#SBATCH --error=/shared/user90/workspace/colossal-ai/ColossalAI/examples/language/llama2/scripts/pretrain/job_%j.err.log
#SBATCH --output=/shared/user90/workspace/colossal-ai/ColossalAI/examples/language/llama2/scripts/pretrain/job_%j.out.log
#SBATCH --time=14-00:00:00

# [90,93-97]
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((RANDOM % 1000 + 25000))
FDR_STR=$(echo "${SLURM_JOB_NAME}" | awk -F'-' '{print $NF}')
LOG_TM=$(date +"%Y%m%d_%H%M")
FDR_STR="mse"
quality=1

echo "SLURM: JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM: JOB_NAME = ${SLURM_JOB_NAME}"
echo "SLURM: NODELIST = ${SLURM_NODELIST}"
echo "SLURM: MASTER_ADDR = $MASTER_ADDR"
echo "SLURM: MASTER_PORT = $MASTER_PORT"
echo "SLURM: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "SLURM: LOG_TM = $LOG_TM"
echo ""

export NCCL_IB_DISABLE=1


cd ../../
srun colossalai run --num_nodes 8 --nproc_per_node 8 pretrain.py \
        --config 7b \
        --dataset cerebras/SlimPajama-627B \
        --batch_size 1 \
        --num_epochs 10 \
        --save_interval 50000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --flash_attention \
        --plugin zero2_cpu \
        --lr 1e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
        
        # --plugin gemini_auto \
        # --flash_attention

#  --hostfile YOUR_HOST_FILE

# --dataset cerebras/SlimPajama-627B
# --dataset togethercomputer/RedPajama-Data-1T-Sample













