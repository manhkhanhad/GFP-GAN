export BASICSR_JIT=True
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1_talkinghead.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1_talkinghead_2.yml --launcher pytorch
# CUDA_LAUNCH_BLOCKING=1