import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
thai = list('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
# print(len(thai))
# url = input('Enter image path to predict : ')
url = 'thaiCharacterReconition/15.png'

# print(chr(3585))

model = keras.Sequential()
model = keras.models.load_model('thaiCharDetecterModel9.keras')
datasets = np.load('thaiDatasets.npy', allow_pickle=True).item()
image1 = (datasets['image'][0])
# model.evaluate(datasets['image'], datasets['answer'])

image = cv2.imread(url, 0)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 3)
image = cv2.resize(image, (30, 30))
# image = np.where(image > 127, 255, 0)
image = cv2.bitwise_not(image)
# ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# image = cv2.blur(image, (3, 3))
# image = image.resize((30, 30))
# np.set_printoptions(threshold=np.inf)
# print(np.array(image))
# cv2.imshow('', image)
plt.imshow(image, cmap='gray')

# image = image[None, :, :]
y_pre = model.predict(image[None, :, :])

print(chr(np.argmax(y_pre) + 3585))
# cv2.waitKey(0)
plt.show()
