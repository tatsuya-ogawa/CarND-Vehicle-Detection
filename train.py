import glob
import matplotlib.image as mpimg
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions import single_img_features
from sklearn.svm import LinearSVC


def train():
    X_scaler = StandardScaler()
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    hist_bins = 32
    svc = LinearSVC()
    car_features = []
    notcar_features = []

    t1 = time.time()
    for fname in glob.glob('train_images/vehicles/*/*.png'):
        image = mpimg.imread(fname)
        feature = single_img_features(image)
        car_features.append(feature)
    for fname in glob.glob('train_images/non-vehicles/*/*.png'):
        image = mpimg.imread(fname)
        feature = single_img_features(image)
        notcar_features.append(feature)
    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to extract HOG features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

train()
