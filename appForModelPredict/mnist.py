from keras._tf_keras.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import datetime

(trainx, trainy), (x, y) = mnist.load_data()
print(type(trainx))

num = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for i in range(trainy.shape[0]):
    a = num[trainy[i]]
    if a < 1000:
        time = datetime.datetime.now()
        time = time.strftime("%d%m%y_%H%M%S")
        # cv2.imshow('', cv2.imread('DatasetHanWritten/1/1.png'))
        path = 'DatasetHanWritten/' + str(45+trainy[i]) + '/' + str(a) + '.png'
        # print(path)
        # cv2.waitKey(0)
        num[trainy[i]] += 1
        cv2.imwrite(path, 255 - trainx[i])
        print(a)

print(num)