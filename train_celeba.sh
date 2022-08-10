set -x

PARTITION=$1
JOB_NAME=$2
WORK_DIR=$3
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:4}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python train.py --curriculum Celeba-slurm --output_dir ${WORK_DIR} --launcher slurm

