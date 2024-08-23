import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras._tf_keras.keras.layers import Dense, Flatten
from keras._tf_keras.keras import layers
import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

dataset = np.load('thaiDatasets.npy', allow_pickle=True).item()

matplotlib.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
matplotlib.rc('font', family='TH Sarabun New')

x, y = dataset['image'], dataset['answer']
# print(np.unique(y))
np.random.seed(12)
shuffle = np.random.permutation(x.shape[0])
# print(shuffle)

x, y = x[shuffle], y[shuffle]
# print(x[0])
# print(y)
ndata = int(x.shape[0]*0.8)

x_train, x_test, y_train, y_test = x[:ndata]/255., x[ndata:]/255., y[:ndata], y[ndata:]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# for index in range(100, 110):
#     plt.imshow(x_train[index])
#     plt.title(dataset['allThaiCharacter'][y_train[index]])
#     plt.show()

model = keras.models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(30, 30, 1)))
model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(Flatten())
# model.add(Flatten(input_shape=[30, 30]))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
y_train = keras.utils.to_categorical(y_train, num_classes=46)
y_test = keras.utils.to_categorical(y_test, num_classes=46)


model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=1)

model.save('thaiCharDetecterModel9.keras', overwrite=True)
# model = keras.models.load_model('thaiCharDetecterModel.keras')

manager = 1456
y_predict = model.predict(x_test, batch_size=16)

model.evaluate(x_test, y_test)
plt.imshow(x_test[manager])
# print(dataset['allThaiCharacter'][(np.argmax(y_predict[manager]))])
plt.title(chr(np.argmax(y_predict)+3585))
plt.show()