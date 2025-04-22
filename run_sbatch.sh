mkdir -p slurm_logs
export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/data/user_data/mprabhud/huggingface_cache
export WANDB_API_KEY=db9454a70e9ee5bfb9cdd83eeb55e2a3ed05c99c


# Patch size 16
sbatch -p 'preempt' --gres=gpu:4 --job-name='16x6' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=16 num_layers=6 exp_group=vary_patchsize_x_num_layers +run_name=16x6" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_6.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_6.err"

sbatch -p 'preempt' --gres=gpu:4 --job-name='16x4' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=16 num_layers=4 exp_group=vary_patchsize_x_num_layers +run_name=16x4" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_4.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_4.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='16x2' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=16 num_layers=2 exp_group=vary_patchsize_x_num_layers +run_name=16x2" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_2.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_16_num_layers_2.err"

# Patch size 32
sbatch -p 'preempt' --gres=gpu:2 --job-name='32x6' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=32 num_layers=6 exp_group=vary_patchsize_x_num_layers +run_name=32x6" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_6.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_6.err"

sbatch -p 'preempt' --gres=gpu:2 --job-name='32x4' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=32 num_layers=4 exp_group=vary_patchsize_x_num_layers +run_name=32x4" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_4.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_4.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='32x2' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=32 num_layers=2 exp_group=vary_patchsize_x_num_layers +run_name=32x2" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_2.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_32_num_layers_2.err"

# Patch size 64
sbatch -p 'preempt' --gres=gpu:1 --job-name='64x6' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=64 num_layers=6 exp_group=vary_patchsize_x_num_layers +run_name=64x6" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_6.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_6.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='64x4' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=64 num_layers=4 exp_group=vary_patchsize_x_num_layers +run_name=64x4" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_4.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_4.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='64x2' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=64 num_layers=2 exp_group=vary_patchsize_x_num_layers +run_name=64x2" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_2.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_64_num_layers_2.err"

# Patch size 128
sbatch -p 'preempt' --gres=gpu:1 --job-name='128x6' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=128 num_layers=6 exp_group=vary_patchsize_x_num_layers +run_name=128x6" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_6.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_6.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='128x4' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=128 num_layers=4 exp_group=vary_patchsize_x_num_layers +run_name=128x4" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_4.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_4.err"

sbatch -p 'preempt' --gres=gpu:1 --job-name='128x2' -x "shire-1-[6,1]" --time=24:00:00 -c8 --mem=50g --wrap="HF_HOME='/data/user_data/mprabhud/huggingface_cache' python train.py patch_size=128 num_layers=2 exp_group=vary_patchsize_x_num_layers +run_name=128x2" --output="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_2.out" --error="slurm_logs/vary_patchsize_x_num_layers_patch_size_128_num_layers_2.err"




