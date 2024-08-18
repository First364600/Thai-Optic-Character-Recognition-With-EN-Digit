import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Flatten, Input, Conv2D
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('thaiorc_v2/BlurNDatasets4.npy', allow_pickle=True).item()
# print(dataset['image'][0].shape)
# plt.imshow(dataset['image'][0], cmap= 'gray')
# plt.show()

model = Sequential()
# model.add(Input(shape=(50, 50)))
model.add(Flatten(input_shape=[28, 28]))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(56, activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
x_train, y_train = dataset['image']/255,dataset['answer']
print(x_train.shape, y_train.shape)
print(np.unique(y_train))
model.fit(x_train, y_train, epochs=10, batch_size=1)
model.save('thaiorc_v2/fullmodel15.keras', overwrite=True)
# print(type(x_train))

# image = cv2.imread('thaiCharacterReconition/11.png', 0)
# image = cv2.resize(image, (50, 50))
# plt.imshow(image)
# image = image[None, :, :]

# pre = model.predict(image)
# print(np.argmax(pre))
# plt.show()