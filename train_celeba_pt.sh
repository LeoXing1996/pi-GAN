python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29823 \
    train.py --curriculum CelebA --launcher torch ${@:3}
