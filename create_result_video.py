import cv2
import os

# video_path = "/home/ldtuan/VideoRestoration/GFPGAN/results/framewise_result/-DiIOp80LOo_0000_S936_E1061_L534_T67_R950_B483"
# video_name = video_path.split('/')[-1]
mode = "cmp"
# input_dir = "/home/ldtuan/VideoRestoration/GFPGAN/results/frame_wise_fullframe_result/raw_result/"
input_dir = "/home/ldtuan/VideoRestoration/GFPGAN/results/frame_wise_finetune2/"
output_dir = "/home/ldtuan/VideoRestoration/GFPGAN/results/frame_wise_finetune2_video_cmp"


# frame_path = os.path.join(video_path, mode)
# output_path = os.path.join(output_dir, video_name + ".mp4")

for i, video_name in enumerate(os.listdir(input_dir)):
    print(i)
    video_path = os.path.join(input_dir, video_name)
    frame_path = os.path.join(video_path, mode)
    output_path = os.path.join(output_dir, video_name + ".mp4")
    cmd = 'ffmpeg -framerate 24 -i '+ frame_path +'/%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p ' + output_path
    print(cmd)
    os.system(cmd)

# size = None
# for image in os.listdir(frame_path):
#     frame = os.path.join(frame_path, image)
#     frame = cv2.imread(frame)

#     if size is None:
#         size = (frame.shape[0], frame.shape[1])
#         out_vid = cv2.VideoWriter(os.path.join(output_dir, video_name + ".mp4"), cv2.VideoWriter_fourcc(*'XVID'), 24.0, size)
#     out_vid.write(frame)
# out_vid.release()

