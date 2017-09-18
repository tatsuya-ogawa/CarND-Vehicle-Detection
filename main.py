import pickle

from moviepy.video.io.VideoFileClip import VideoFileClip
from functions import *

tmp_dic = {
    "heatmap_list": [],
    "dist_pickle": pickle.load(open("svc_pickle.p", "rb"))
}


def process_frame(src_image):
    heatmap_list = tmp_dic["heatmap_list"]
    dist_pickle = tmp_dic["dist_pickle"]
    result, heatmap_list, _, _, _ = process_image(src_image, dist_pickle, heatmap_list)
    tmp_dic["heatmap_list"] = heatmap_list
    return result


def main():
    video_name = 'project_video.mp4'
    clip = VideoFileClip(video_name)

    clip = clip.fl_image(process_frame)
    video_output = 'project_out.mp4'
    clip.write_videofile(video_output, audio=False)


main()
