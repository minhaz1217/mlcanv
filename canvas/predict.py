import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import os
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def predictThis():
    charMap = ["ek", "dui", "tin", "char", "pach", "choy", "saat", "aat", "noy", "shunno", "shore-o", "shore-a",
               "ro-shui", "dhir-ghoi", "ro-shau", "dhir-ghau", "ri", "a", "oi", "oo", "ou", "ka", "kkha", "go", "gho",
               "uo", "co", "cho", "borgiarjo", "zho", "eio", "tto", "ttho", "ddo", "ddho", "dd-no", "to", "tho", "do",
               "dho", "d-no", "po", "fo", "bo", "vo", "mo", "jo", "ro", "lo", "talobbo-sho", "murdhonno-sho",
               "donte-sho", "ho", "d-ro", "dshunno-ro", "ontestio", "khandoto", "onnushar", "bissorgo", "chandrebindu"]
    bcharMap = ["১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "০", " অ", " আ", " ই", " ঈ", " উ", " ঊ", " ঋ", " এ", " ঐ",
                " ও", " ঔ", " ক", " খ", " গ", "  ঘ", " ঙ", " চ", " ছ", " জ", " ঝ", " ঞ", " ট", " ঠ", " ড", "ঢ", " ণ",
                " ত", " থ", " দ", " ধ", "ন", " প", " ফ", " ব", " ভ", " ম", " য", " র", " ল", " শ", " ষ", " স", " হ",
                " ড়", " ঢ়", " য়", " ৎ", "ং", " ঃ", " ঁ"]
    MODEL_NAME1 = "bangla-0.001_6conv_grayep500_ts150_train_data_r50_denoised_minus255_gray_27._28-8-2017_6-44"
    MODEL_NAME2 = "bangla-0.001_6conv-basic_s100_grayep500_train_data_r50grag_27._28-8-2017_4-59"
    MODEL_NAME3 = "bangla-0.001-6conv-basic_s100_gray_train_data_r50_gray_5.npy_27-8-2017_7-9"
    MODEL_NAME4 = "bangla-0.001-6conv-basic_s100_gray_train_data_r50_gray_5.npy_27-8-2017_7-10"
    MODEL_NAME = MODEL_NAME1

    fileDir = "C:\/tflow\/bangla\/train_data\/test\/"
    file = "37_002.png"
    fileName = "mine.png"
    originalFileDirectory = "C:\/tflow\/bangla\/train_data\/test\/orig\/"
    IMG_SIZE = 50
    image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = 255 - image
    #cv2.imshow("", image)
    #cv2.waitKey(0)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE ))
    loadedImage = image

    #print(MODEL_NAME)
    tf.reset_default_graph()
    myOutputSize = 60  # outputsize
    LR = 1e-3
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, myOutputSize,
                              activation='softmax')  # it was 3 previously 60 is the size of one hot array
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log')
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        #print("MODEL LOADED")
        p = model.predict([loadedImage.reshape(IMG_SIZE, IMG_SIZE, 1)])[0]
        predictNumber = np.argmax(p)
        print(charMap[predictNumber])
        return bcharMap[predictNumber]
    else:
        return -1


#print(predictThis())