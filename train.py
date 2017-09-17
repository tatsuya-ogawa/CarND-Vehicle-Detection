import glob
import matplotlib.image as mpimg
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions import single_img_features
from sklearn.svm import LinearSVC


def train(color_space, orient, cell_per_block, pix_per_cell, spatial_size, hist_bins, hog_channel,
          spatial_feat=True, hist_feat=True, hog_feat=True):
    svc = LinearSVC()
    car_features = []
    notcar_features = []

    t1 = time.time()
    cars = [fname for fname in glob.glob('train_images/vehicles/**/*.png')]
    notcars = [fname for fname in glob.glob('train_images/non-vehicles/**/*.png')]
    # sample_size = min(len(cars), len(notcars))
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    for fname in cars:
        image = mpimg.imread(fname)
        feature = single_img_features(image, color_space=color_space, orient=orient, cell_per_block=cell_per_block,
                                      pix_per_cell=pix_per_cell,
                                      spatial_size=spatial_size, hist_bins=hist_bins, hog_channel=hog_channel,
                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        car_features.append(feature)

    for fname in notcars:
        image = mpimg.imread(fname)
        feature = single_img_features(image, color_space=color_space, orient=orient, cell_per_block=cell_per_block,
                                      pix_per_cell=pix_per_cell,
                                      spatial_size=spatial_size, hist_bins=hist_bins, hog_channel=hog_channel,
                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features.append(feature)

    t2 = time.time()
    extract_time = round(t2 - t1, 2)
    print(extract_time, 'Seconds to extract features...')

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
    training_time = round(t2 - t1, 2)
    print(training_time, 'Seconds to train SVC...')
    score = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', score)

    src_pickle = {"svc": svc, "scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell,
                  "cell_per_block": cell_per_block, "spatial_size": spatial_size, "hist_bins": hist_bins,
                  "hog_channel": hog_channel, "color_space": "YCrCb"}
    return src_pickle, score, extract_time, training_time


def train_and_save():
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    hist_bins = 32
    hog_channel = 'ALL'
    color_space = 'YCrCb'
    result_pickle, _, _, _ = train(color_space=color_space, orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   spatial_size=spatial_size, hist_bins=hist_bins, hog_channel=hog_channel)
    print("Saving pickle")
    pickle.dump(result_pickle, open("svc_pickle.p", "wb"))


def train_iterate():
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    color_space = 'YCrCb'

    result_list = []
    spatial_size = (32, 32)
    hist_bins = 32
    color_space_list = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
    pix_per_cell_list = [4, 8, 12, 16]
    cell_per_block_list = [1, 2, 3]
    hog_channel_list = [0, 1, 2, 'ALL']
    orient_list = [6, 7, 9, 11, 12]

    result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                              pix_per_cell=pix_per_cell,
                                                              cell_per_block=cell_per_block,
                                                              spatial_size=spatial_size, hist_bins=hist_bins,
                                                              hog_channel=hog_channel, spatial_feat=False)
    result_list.append({
        "score": score,
        "extract_time": extract_time,
        "training_time": training_time,
        "picle": result_pickle
    })

    result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                              pix_per_cell=pix_per_cell,
                                                              cell_per_block=cell_per_block,
                                                              spatial_size=spatial_size, hist_bins=hist_bins,
                                                              hog_channel=hog_channel, hist_feat=False)
    result_list.append({
        "score": score,
        "extract_time": extract_time,
        "training_time": training_time,
        "picle": result_pickle
    })

    result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                              pix_per_cell=pix_per_cell,
                                                              cell_per_block=cell_per_block,
                                                              spatial_size=spatial_size, hist_bins=hist_bins,
                                                              hog_channel=hog_channel, hog_feat=False)
    result_list.append({
        "score": score,
        "extract_time": extract_time,
        "training_time": training_time,
        "picle": result_pickle
    })

    for _color_space in color_space_list:
        result_pickle, score, extract_time, training_time = train(color_space=_color_space, orient=orient,
                                                                  pix_per_cell=pix_per_cell,
                                                                  cell_per_block=cell_per_block,
                                                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                                                  hog_channel=hog_channel)
        result_list.append({
            "score": score,
            "extract_time": extract_time,
            "training_time": training_time,
            "picle": result_pickle
        })

    for _pix_per_cell in pix_per_cell_list:
        result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                                  pix_per_cell=_pix_per_cell,
                                                                  cell_per_block=cell_per_block,
                                                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                                                  hog_channel=hog_channel)
        result_list.append({
            "score": score,
            "extract_time": extract_time,
            "training_time": training_time,
            "picle": result_pickle
        })

    for _cell_per_block in cell_per_block_list:
        result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                                  pix_per_cell=pix_per_cell,
                                                                  cell_per_block=_cell_per_block,
                                                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                                                  hog_channel=hog_channel)
        result_list.append({
            "score": score,
            "extract_time": extract_time,
            "training_time": training_time,
            "picle": result_pickle
        })

    for _hog_channel in hog_channel_list:
        result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=orient,
                                                                  pix_per_cell=pix_per_cell,
                                                                  cell_per_block=cell_per_block,
                                                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                                                  hog_channel=_hog_channel)
        result_list.append({
            "score": score,
            "extract_time": extract_time,
            "training_time": training_time,
            "picle": result_pickle
        })

    for _orient in orient_list:
        result_pickle, score, extract_time, training_time = train(color_space=color_space, orient=_orient,
                                                                  pix_per_cell=pix_per_cell,
                                                                  cell_per_block=cell_per_block,
                                                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                                                  hog_channel=hog_channel)
        result_list.append({
            "score": score,
            "extract_time": extract_time,
            "training_time": training_time,
            "picle": result_pickle
        })
    print("Saving pickle")
    pickle.dump(result_list, open("svc_pickle_list.p", "wb"))


train_iterate()
