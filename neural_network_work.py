from __future__ import division, print_function, absolute_import
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageOps
import PIL.ImageOps
import tflearn
import os
import cv2
from networkx.drawing.tests.test_pylab import plt
import seaborn as sn  # heatmap
from tflearn.data_utils import to_categorical
import tflearn.data_utils as du
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from resizeimage import resizeimage
from tqdm import tqdm


def confusion_matrix(y_pred, y_real, DT_predict_prob):
    array_different_letters = []
    array_different_letters.append(y_pred[0])
    for i in range(1, len(y_pred)):
        if y_pred[i] not in array_different_letters:
            array_different_letters.append(y_pred[i])

    for i in range(len(y_real)):
        if y_real[i] not in array_different_letters:
            array_different_letters.append(y_real[i])

    c_m_len = np.max(array_different_letters) + 1
    confusion_mat = np.zeros((c_m_len, c_m_len))
    max_predict_prob = []
    for i in range(len(y_real)):
        if max(DT_predict_prob[i]) > 0.6:
            confusion_mat[y_real[i]][y_pred[i]] += 1

    heatmap = plt.axes()
    heatmap = sn.heatmap(confusion_mat, annot=True, fmt='g', cmap="Blues")
    heatmap.set_title('confusion_matrix')
    plt.savefig("final_result.png")


def sort_internet_dataset(path):
    return int(path.split('_')[1].split('_')[0])


def sort_our_dataset(path):
    return int(path.split('_')[1].split('.')[0])


if __name__ == "__main__":
    # train 1 - dataset from internet
    trainx = []
    path_train = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/Train_5_letters")):
        path = os.path.join("C:/Users/aviga/Desktop/Train_5_letters", img)
        path_train.append(path)

    sort_path_train = sorted(path_train, key=sort_internet_dataset)
    for path in path_train:
        img = Image.open(path)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        trainx.append(img)

    trainx = np.asarray(trainx)
    trainx = trainx.reshape([-1, 32, 32, 1])
    trainx, mean1 = du.featurewise_zero_center(trainx)

    # train_y
    trainy = pd.read_csv("C:/Users/aviga/Desktop/y_Train_5_letters.csv", header=None)
    trainy = trainy.values.astype('int32') - 1
    trainy = to_categorical(trainy, 30)

    # train2
    trainx2 = []
    path_train_2 = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/train")):
        path = os.path.join("C:/Users/aviga/Desktop/train", img)
        path_train_2.append(path)

    sort_path_train_2 = sorted(path_train_2, key=sort_our_dataset)
    for path in sort_path_train_2:
        img = Image.open(path)
        img = PIL.ImageOps.invert(img)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        trainx2.append(img)

    trainx2 = np.asarray(trainx2)
    trainx2 = trainx2.reshape([-1, 32, 32, 1])
    trainx2, mean3 = du.featurewise_zero_center(trainx2)

    # train2_y
    trainy2 = pd.read_csv("C:/Users/aviga/Desktop/train.csv", header=None)
    trainy2 = trainy2.values.astype('int32') - 1
    trainy2 = to_categorical(trainy2, 30)

    # test
    testx = []
    path_test = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/test")):
        path = os.path.join("C:/Users/aviga/Desktop/test", img)
        path_test.append(path)

    sort_path_test = sorted(path_test, key=sort_our_dataset)
    for path in sort_path_test:
        img = Image.open(path)
        img = PIL.ImageOps.invert(img)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        testx.append(img)

    testx = np.asarray(testx)
    testx = testx.reshape([-1, 32, 32, 1])
    testx, mean2 = du.featurewise_zero_center(testx)

    # test_y
    testy = pd.read_csv("C:/Users/aviga/Desktop/test.csv", header=None)
    testy_before = testy.values.astype('int32') - 1
    testy = to_categorical(testy_before, 30)

    # Building convolutional network
    network = input_data(shape=[None, 32, 32, 1], name='input')
    network = conv_2d(network, 80, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 30, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    # model complile
    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.fit({'input': trainx}, {'target': trainy}, n_epoch=60,
              snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

    model.fit({'input': trainx2}, {'target': trainy2}, n_epoch=180,
              snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

    # Evaluate model
    score = model.evaluate(testx, testy)
    print('Test accuarcy: %0.2f%%' % (score[0] * 100))

    DT_predict_prob = model.predict(testx)  # Predictions on Testing data
    np.set_printoptions(threshold=sys.maxsize, formatter={'float_kind':'{:f}'.format})

    DT_predict = model.predict(testx)
    A = np.squeeze(np.asarray(testy_before))
    print(A)
    classes = np.argmax(DT_predict, axis=1)
    print(classes)

    confusion_matrix(classes, A, DT_predict_prob)
