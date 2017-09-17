import cv2

from functions import get_hog_features


def display_image(image, file_name=None):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    for channel in range(image.shape[2]):
        hog = get_hog_features(image[:,:,channel])
