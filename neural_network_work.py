from __future__ import division, print_function, absolute_import

# from keras.backend.tensorflow_backend import set_session
import glob
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
from scipy import ndimage
from sklearn.metrics import confusion_matrix
from sympy.printing.tests.test_tensorflow import tf
from xlwt import Workbook

from tensorflow.python.ops.summary_ops_v2 import graph
from tflearn.data_utils import to_categorical
import tflearn.data_utils as du
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tensorflow import keras
from resizeimage import resizeimage
from keras.models import load_model
from tqdm import tqdm


def check(path):
    return int(path.split('_')[1].split('_')[0])
    # return s[0]


if __name__ == "__main__":
    # check('C:/Users/aviga/Desktop/train\\id_1000_label_13.png')
    trainx = []
    path_array_train = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/train")):
        path = os.path.join("C:/Users/aviga/Desktop/train", img)
        path_array_train.append(path)
    # print(path_array)

    sort_list = sorted(path_array_train, key=check)
    print("%%%%%%%%%%%%%%%%%%%%%%")
    print(sort_list)
    # print(len(path_array_train))
    for path in sort_list:
        img = Image.open(path)
        img = ImageOps.mirror(img)
        img = np.rot90(img, 1)
        img = Image.fromarray(img)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        trainx.append(img)

    trainx = np.asarray(trainx)
    trainx = trainx.reshape([-1, 32, 32, 1])

    trainx, mean1 = du.featurewise_zero_center(trainx)

    trainy = pd.read_csv("csvTrainLabel 13440x1.csv", header=None)
    trainy = trainy.values.astype('int32') - 1
    for i in range(len(trainy)):
        if trainy[i][0] == 0:
            print(i)
            print()
            # print(trainy[i][0])
            # print("C:/Users/aviga/Documents/תואר הנדסת תוכנה/שנה ד/פרוייקט גמר/test/alif/myphoto" + str(i) + ".jpeg")
            cv2.imwrite("alif/myphoto" + str(i) + ".png", trainx[i])
            # trainx[i].save("C:/Users/aviga/Documents/תואר הנדסת תוכנה/שנה ד/פרוייקט גמר/test/alif/myphoto" + str(i) + ".jpeg", "JPEG")

    trainy = to_categorical(trainy, 28)
    # print(trainy)

    testx = []
    path_array_test = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/mim")):
        path = os.path.join("C:/Users/aviga/Desktop/mim", img)
        path_array_test.append(path)

    # sort_list_2 = sorted(path_array_test, key=check)
    # print(path_array_test)
    # print(len(path_array_test))
    for path in path_array_test:
        img = Image.open(path)
        img = PIL.ImageOps.invert(img)
        # show(img)
        # cv2.waitKey(0)
        img = ImageOps.mirror(img)
        img = np.rot90(img, 1)
        img = Image.fromarray(img)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        testx.append(img)
    testx = np.asarray(testx)
    testx = testx.reshape([-1, 32, 32, 1])
    testx, mean2 = du.featurewise_zero_center(testx)

    # print(testx.shape)
    cv2.imshow("avug", testx[10])
    cv2.waitKey(0)
    testy = pd.read_csv("mim_y.csv", header=None)
    # print(testy.shape)
    testy = testy.values.astype('int32') - 1

    testy = to_categorical(testy, 28)

    # train2
    trainx2 = []
    path_array_train_2 = []
    for img in tqdm(os.listdir("C:/Users/aviga/Desktop/mim_train")):
        path = os.path.join("C:/Users/aviga/Desktop/mim_train", img)
        path_array_train_2.append(path)

    for path in path_array_train_2:
        img = Image.open(path)
        img = PIL.ImageOps.invert(img)
        # show(img)
        # cv2.waitKey(0)
        img = ImageOps.mirror(img)
        img = np.rot90(img, 1)
        img = Image.fromarray(img)
        img = resizeimage.resize_cover(img, [32, 32])
        img = np.asarray(img)
        trainx2.append(img)
    trainx2 = np.asarray(trainx2)
    trainx2 = trainx2.reshape([-1, 32, 32, 1])
    trainx2, mean3 = du.featurewise_zero_center(trainx2)

    # print(testx.shape)
    cv2.imshow("avug", trainx2[10])
    cv2.waitKey(0)
    trainy2 = pd.read_csv("mim_train_y.csv", header=None)
    # print(testy.shape)
    trainy2 = trainy2.values.astype('int32') - 1

    trainy2 = to_categorical(trainy2, 28)

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
    network = fully_connected(network, 28, activation='softmax')
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    # model complile
    model = tflearn.DNN(network, tensorboard_verbose=0)
    # model fitting
    model.fit({'input': trainx}, {'target': trainy}, n_epoch=40,
              # validation_set=({'input': testx}, {'target': testy}),
              snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

    model.fit({'input': trainx2}, {'target': trainy2}, n_epoch=10,
              # validation_set=({'input': testx}, {'target': testy}),
              snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

    # model.save("my_model.h5")
    #
    # model = load_model('my_model.h5.index')

    # Evaluate model
    score = model.evaluate(testx, testy)
    print('Test accuarcy: %0.2f%%' % (score[0] * 100))

    # DT_predict = model.predict(testx)  # Predictions on Testing data
    # np.set_printoptions(threshold=sys.maxsize, formatter={'float_kind':'{:f}'.format})
    # print(DT_predict)

    DT_predict = model.predict(testx)
    print(DT_predict)

    # heatmap = plt.axes()
    # confusion_matrix = confusion_matrix(testy, DT_predict)
    # heatmap = sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap="Blues")
    # heatmap.set_title('nun_confusion_matrix')
    # plt.show()

    # wb = Workbook()
    # # add_sheet is used to create sheet
    # sheet1 = wb.add_sheet("mim")
    #
    # # Create the titles of the columns
    # for i in range(1, 29):
    #     sheet1.write(i, 0, str(i))
    #     sheet1.write(0, i, "class " + str(i))
    #
    # # Add the points
    # for i in range(1, len(DT_predict)):
    #     for j in range(1, len(DT_predict[0])):
    #         sheet1.write(i, j, str(f"{DT_predict[i-1][j-1]:.9f}"))
    #
    # wb.save('probability_mim.xls')