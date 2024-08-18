import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
import cv2
from characterDetection import character_detection

output = str("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789")
model = keras.models.Sequential()
model = keras.models.load_model('data/fullmodel15.keras')

def Predict():
    image = cv2.imread('data/image.png', 0)    
    image = character_detection(image, (28, 28))
    pre = model.predict(image[None, :, :])
    return output[np.argmax(pre) - 1]