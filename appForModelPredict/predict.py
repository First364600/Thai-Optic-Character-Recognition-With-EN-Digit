import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from characterDetection import character_detection

output = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789'
print(keras.__version__)

  model = keras.models.Sequential()
model = keras.models.load_model(r'appForModelPredict/data/class_bbox5.h5')
  model.summary()
  model.compile(
      optimizer='adam', 
      loss={'class': 'categorical_crossentropy', 'bbox': 'mse'}, 
      metrics={'class': 'accuracy', 'bbox': 'accuracy'}
  )

def calculate_confidence(predictions):
      Calculate the probability for each class prediction
    confidence_scores = tf.reduce_max(predictions, axis=-1)    Max probability as confidence score per time step
      print(confidence_scores)
    avg_confidence = tf.reduce_mean(confidence_scores)         Mean confidence for the whole sequence
    return avg_confidence.numpy() * 100 

def Predict():
    image = cv2.imread('appForModelPredict/data/image.png', 0)
      saveImage(image, '1.png', 'DatasetHanWritten/other')
    
      image = cv2.resize(image, (50, 50))
      image = character_detection(image, (254, 254))
    image = 255 - cv2.resize(image, (254, 254))

    pre = model.predict(image[None, :, :])
    clss, bbox = pre
      print(bbox[0])
    x1, y1, x2, y2 = np.array(bbox[0], dtype='uint8') * 254
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('', image)
    score = calculate_confidence(clss)
      cv2.imshow('', image)
    
      cv2.waitKey(0)
    return output[np.argmax(clss)] + '  ( ' + str(round(score, 2)) + ' % )'