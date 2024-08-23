import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from characterDetection import character_detection

output = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789'

model = keras.models.Sequential()
model = keras.models.load_model('data/fullmodel16.keras')

def Predict():
    image = cv2.imread('data/image.png', 0)
    # saveImage(image, '1.png', 'DatasetHanWritten/other')
    
    # image = cv2.resize(image, (50, 50))
    image = character_detection(image, (28, 28))
    pre = model.predict(image[None, :, :])
    cv2.imshow('', image)
    
    cv2.waitKey(0)
    return output[np.argmax(pre) - 1]