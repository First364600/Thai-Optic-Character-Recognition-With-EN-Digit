
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras._tf_keras.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
fashio_mnist = keras.datasets.fashion_mnist
mnist = keras.datasets.mnist

# print(mnist.load_data())
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_test))
# print(type(mnist.load_data()))

x_train = x_train/255
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
model = keras.Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

model.save('mark.keras', overwrite=True)
_model = keras.models.load_model('mark.keras')
predictions = model.predict([x_test], batch_size=64)
predictions[0]

print(np.argmax(predictions[0]))
