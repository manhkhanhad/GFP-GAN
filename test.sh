export BASICSR_JIT=True
CUDA_VISIBLE_DEVICES=1 python inference_gfpgan_all_video.py -i "/home/ldtuan/VideoRestoration/dataset/TalkingHead/degradation/" -o "/home/ldtuan/VideoRestoration/GFPGAN/results/frame_wise_finetune2/" -v v1_finetune_256 -s 1 --model_path "/home/ldtuan/VideoRestoration/GFPGAN/experiments/finetune2_GFPGAN_TalkingHead/models/net_g_420000.pth" 