#!/bin/bash -l

#SBATCH --job-name vlr
#SBATCH --partition preempt
#SBATCH --time 2-00:00:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:L40S:4
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 6G
#SBATCH --signal B:USR1@120
#SBATCH --requeue

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
export NCCL_P2P_DISABLE=1   # Don't use NV-Link (causes hangs on some systems).

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

timestamp() {
    date "+%y-%m-%d %H:%M:%S"
}

echo "$(timestamp) sbatch script started."

handle_signal() {
    master_pid=$(cat .master.pid.$SLURM_JOB_ID)
    echo "$(timestamp) sbatch script got signal $1; forwarding to process $master_pid on node $MASTER_ADDR."
    kill -s $1 $master_pid
}

for signal in INT USR1; do
    trap "handle_signal $signal; wait" $signal
done

patches_per_side=(3 10 14 18 22 25 27 30 32)
crop_px=$((27 * 16))

conda activate vlrh

srun ./torchrun.sh &
wait

echo "$(timestamp) sbatch script finished."
