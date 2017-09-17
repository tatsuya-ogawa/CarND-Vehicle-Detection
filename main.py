import pickle

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label
from functions import *

tmp_dic = {
    "heatmap_list": [],
    "dist_pickle": pickle.load(open("svc_pickle.p", "rb"))
}


def process_image(src_image, file_name=None):
    scales = [1.5, 2.0]
    y_starts = [400, 400]
    y_stops = [556, 656]
    dist_pickle = tmp_dic["dist_pickle"]

    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    heatmap_weight = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float)

    heatmap_list = tmp_dic["heatmap_list"]
    box_list = []
    for scale, ystart, ystop in zip(scales, y_starts, y_stops):
        box_list_tmp = find_cars(src_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                 cell_per_block,
                                 spatial_size,
                                 hist_bins)
        box_list.extend(box_list_tmp)

    heat = np.zeros_like(src_image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, 1)
    current_heatmap = np.clip(heat, 0, 255)

    heatmap_list.append(current_heatmap)
    heatmap_list = heatmap_list[-min(len(heatmap_weight), len(heatmap_list)):]
    heatmap = np.sum(
        np.array(heatmap_list, dtype=np.float) * heatmap_weight[-len(heatmap_list):].reshape((-1, 1, 1)),
        axis=0)
    heatmap = apply_threshold(heatmap, sum(heatmap_weight[-len(heatmap_list):]) * 2)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(src_image), labels)

    tmp_dic["heatmap_list"] = heatmap_list
    return draw_img


def main():
    video_name = 'project_video.mp4'
    clip = VideoFileClip(video_name)

    clip = clip.fl_image(process_image)
    video_output = 'project_out.mp4'
    clip.write_videofile(video_output, audio=False)


main()
