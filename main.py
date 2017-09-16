import pickle

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label
from functions import *


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img,
              ystart, ystop, scale,
              svc, X_scaler, orient,
              pix_per_cell, cell_per_block,
              spatial_size, hist_bins):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(X)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box_list.append(((xbox_left, ytop_draw + ystart),
                                 (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return box_list


def process_images():
    # cap = cv2.VideoCapture(video_name)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20, (1280, 720))
    # ystart = 400
    # ystop = 656
    tmp_dic = {
        "heatmap_list": [],
        "dist_pickle": pickle.load(open("svc_pickle.p", "rb"))
    }

    def process_image(src_image):
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
        # for box in box_list:
        #     cv2.rectangle(draw_img, box[0],
        #                   box[1], (255, 0, 0), 6)
        # cv2.rectangle(draw_img, (0, 400),
        #               (1280, 656), (0, 255, 0), 6)
        # cv2.rectangle(draw_img, (0, 400),
        #               (1280, 464), (0, 255, 0), 6)

        # out.write(draw_img)
        tmp_dic["heatmap_list"] = heatmap_list
        return draw_img


def main():
    video_name = 'project_video.mp4'
    clip = VideoFileClip(video_name)

    clip = clip.fl_image(process_images)
    video_output = 'project_out.mp4'
    clip.write_videofile(video_output, audio=False)


main()
