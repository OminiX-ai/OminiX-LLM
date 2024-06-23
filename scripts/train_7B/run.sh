# export CUDA_LAUNCH_BLOCKING=1

# HOSTFILE=$(realpath hosts.txt)
# --hostfile hosts.txt 

colossalai run --nproc_per_node 1 pretrain.py \
        --config 7b \
        --dataset togethercomputer/RedPajama-Data-1T-Sample \
        --batch_size 1 \
        --num_epochs 5 \
        --save_interval 5000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --plugin zero2_cpu \
        --lr 2e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
        #--expanded_model /shared/user90/workspace/llama-factory/LLaMA-Factory/models/mistral-v0.2-7b-36layers \
        #--load /home/user1/workspace/colossal-ai/ColossalAI/examples/language/llama2/output-checkpoints/20240506-epoch2-step50000\
        # --expanded_model /home/user1/workspace/llama-factory/LLaMA-Factory/models/mistral-v0.2-7b-36layers \
        # --expanded_model /home/user1/workspace/llama-factory/LLaMA-Factory/models/mistral-v0.2-7b-36layers \
        # --load /home/user1/workspace/llama-factory/LLaMA-Factory/models/mistral-v0.2-7b-36layers \
        # --load /home/user1/workspace/llama-pro/LLaMA-Pro/checkpoints/Mistral-7B-Instruct-v0.2 \

        
        # --plugin gemini_auto \
        # --flash_attention

#  --hostfile YOUR_HOST_FILE

# --dataset cerebras/SlimPajama-627B
# --dataset togethercomputer/RedPajama-Data-1T-Sample

