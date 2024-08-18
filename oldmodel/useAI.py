import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import easyocr.recognition
import keras
import easyocr
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model1 = keras.Sequential()
model1 = keras.models.load_model('mark.keras')
model1.evaluate(x_test, y_test)
image = 6000

image1 = Image.open('AIFORTHAI-ThaiOCRCorpus\ThaiOCR\ThaiOCR-TrainigSet\Thai\Thai - Copy/161/161 (1).bmp')
image1 = image1.resize((28, 28))
image1 = np.array(image1, dtype='float32')
image1 = np.expand_dims(image1, axis=0)
print('this is ', image1.shape)
# plt.imshow(image1)
# plt.show()
# print(x_test[0].shape)
# Sample image with dimensions (9, 7)

# x = (x_test[image])[None, :, :]/255
# x = np.expand_dims(x_test[image], axis=0)
predict = model1.predict(image1)
# print(x_test[0].shape)
# plt.imshow(image1, cmap='binary')
# plt.title(np.argmax(predict))
# plt.show()

print(np.argmax(predict))
