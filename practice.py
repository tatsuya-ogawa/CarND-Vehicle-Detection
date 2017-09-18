import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

from functions import get_hog_features, slide_window, process_image
import pickle


def display_hog_image(image, file_name=None):
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    f, axs = plt.subplots(2, 4, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    axs = axs.ravel()
    index = 0

    axs[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[index].axis('off')
    axs[index].set_title('Original Image', fontsize=20)
    index = index + 1

    for channel in range(image_YCrCb.shape[2]):
        axs[index].axis('off')
        axs[index].set_title('YCrCb channel {}'.format(channel), fontsize=20)
        axs[index].imshow(image_YCrCb[:, :, channel], cmap='gray')
        index = index + 1

    hog_visualizes = []
    for channel in range(image_YCrCb.shape[2]):
        hog, vis = get_hog_features(image_YCrCb[:, :, channel], orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, vis=True)
        hog_visualizes.append(vis)

    hog_visualize = np.dstack(hog_visualizes)
    axs[index].imshow(hog_visualize)
    axs[index].set_title('Hog summary', fontsize=20)
    index = index + 1

    for vis in hog_visualizes:
        axs[index].imshow(vis, cmap='gray')
        axs[index].set_title('Hog channel {}'.format(channel), fontsize=20)
        index = index + 1
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def display_images(file_names, shape, title, output_path=None):
    f, axs = plt.subplots(shape[0], shape[1], figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    axs = axs.ravel()
    for index, file_name in enumerate(file_names):
        image = cv2.imread(file_name)
        axs[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[index].axis('off')
        axs[index].set_title(title, fontsize=20)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def display_train_images():
    sample_count = np.array([2, 5], dtype=np.int32)
    cars = [fname for fname in glob.glob('train_images/vehicles/**/*.png')]
    cars = np.random.choice(cars, np.prod(sample_count))
    display_images(cars, sample_count, 'car', 'output_images/cars.png')

    notcars = [fname for fname in glob.glob('train_images/non-vehicles/**/*.png')]
    notcars = np.random.choice(notcars, np.prod(sample_count))
    display_images(notcars, sample_count, 'not car', 'output_images/notcars.png')


def display_hog():
    cars = [fname for fname in glob.glob('train_images/vehicles/**/*.png')]
    car = np.random.choice(cars, 1)[0]
    image = cv2.imread(car)
    display_hog_image(image, 'output_images/car_hog.png')

    notcars = [fname for fname in glob.glob('train_images/non-vehicles/**/*.png')]
    notcar = np.random.choice(notcars, 1)[0]
    image = cv2.imread(notcar)
    display_hog_image(image, 'output_images/notcar_hog.png')


def list_results():
    svc_results = pickle.load(open("svc_pickle_list.p", "rb"))
    svc_results.sort(key=lambda x: -x["score"])
    with open('list.txt', 'w') as f:
        columns = " Use Spatial | Use Histogram | Use Hog | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Extract Time | Training Time | Score ".split(
            "|")
        f.write(
            "|" + "|".join(columns) + "|\r"
        )
        f.write(
            "|" + "|".join([" :-----------------: " for column in columns]) + "|\r"
        )
        f.writelines(
            [
                "|" + "|".join([str(item) for item in [
                    'True' if result['spatial_feat'] else 'False',
                    'True' if result['hist_feat'] else 'False',
                    'True' if result['hog_feat'] else 'False',
                    result['pickle']['color_space'],
                    result['pickle']['orient'],
                    result['pickle']['pix_per_cell'],
                    result['pickle']['cell_per_block'],
                    result['pickle']['hog_channel'],
                    result['extract_time'],
                    result['training_time'],
                    result['score']
                ]]) + "|\r"
                for result in svc_results]
        )


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def sliding_window_image():
    scales = [0.8, 1.5, 2.0]
    y_starts = [400, 400, 400]
    y_stops = [480, 556, 656]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    image = cv2.imread('test_images/test1.jpg')
    for scale, y_start, y_stop, color in zip(scales, y_starts, y_stops, colors):
        boxes = slide_window(image, y_start_stop=[y_start, y_stop], xy_window=[64 * scale, 64 * scale])
        image = draw_boxes(image, boxes, color)
    cv2.imwrite('output_images/sliding_window1.jpg', image)


def frames():
    def colorscale(v):
        t = np.math.cos(4 * np.math.pi * v)
        c = int(((-t / 2) + 0.5) * 255)
        if v >= 1.0:
            return (255, 0, 0)
        elif v >= 0.75:
            return (255, c, 0)
        elif v >= 0.5:
            return (c, 255, 0)
        elif v >= 0.25:
            return (0, 255, c)
        elif v >= 0:
            return (0, c, 255)
        else:
            return (0, 0, 255)
    video_name = 'project_video.mp4'
    clip = VideoFileClip(video_name).subclip(38, 40)
    iterator = clip.iter_frames()
    p = pickle.load(open("svc_pickle.p", "rb"))
    f, axs = plt.subplots(6, 4, figsize=(20, 20))
    f.subplots_adjust(hspace=0.8, wspace=0.2)
    axs = axs.ravel()
    for i in range(0, 24, 4):
        image = iterator.__next__()
        result, _, boxes, heatmap, labels = process_image(image, p)
        image = draw_boxes(image, boxes)
        axs[i].imshow(image)
        axs[i].set_title('frame {}'.format(i))

        zeros = np.zeros_like(heatmap)
        #heatmap = np.dstack((heatmap, zeros, zeros))
        axs[i + 1].imshow(heatmap*16)
        axs[i + 1].set_title('heatmap {}'.format(i))

        axs[i + 2].imshow(np.sum(labels,axis=0)*16,cmap='gray')
        axs[i + 2].set_title('label {}'.format(i))

        axs[i + 3].imshow(result)
        axs[i + 3].set_title('result {}'.format(i))


    plt.savefig('./output_images/heatmap_output.jpg')


def detect_output():
    p = pickle.load(open("svc_pickle.p", "rb"))
    f, axs = plt.subplots(3, 2)
    f.subplots_adjust(hspace=0.8, wspace=0.2)
    axs = axs.ravel()
    for i in range(1, 7):
        image = cv2.imread('./test_images/test{}.jpg'.format(i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, _, boxes, _, _ = process_image(image, p)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = draw_boxes(image, boxes)
        axs[i - 1].imshow(image)
        axs[i - 1].set_title('test{}.jpg'.format(i))
    plt.savefig('./output_images/detect_output.jpg')

