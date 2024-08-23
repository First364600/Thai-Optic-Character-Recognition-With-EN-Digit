import cv2
import numpy as np
from characterDetection import *
import os
import matplotlib.pyplot as plt



image1, answer = [], []
i = 0

for n in range(1, 55):
        image_dir = 'DatasetHanWritten/' + str(n)
        # if n!= 13:
        #         continue
        # print(n)
        for fname in os.listdir(image_dir):
                if fname.lower().endswith(('png')):
                        # if fname == '408.png':
                        # url = 'DatasetHanWritten/' + str(n) + '/' + str(j) + '.png'
                        url = os.path.join(image_dir, fname)
                        image = cv2.imread(url, 0)
                        image = character_detection(image, (28, 28))
                        # print(url)
                        # print(image.shape)
                        # cv2.imshow('',image)
                        # cv2.waitKey(0)
                        # plt.imshow(image, cmap='binary')
                        # plt.show()
                        image1.append(np.asarray(image).astype('float64'))
                        answer.append(np.asarray(n).astype('float64'))
                        i += 1
print(i)
data = {'image' : np.asarray(image1),
        'answer' : np.asarray(answer),
        'output' : 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789'}

np.save('thaiorc_v2/BlurNDatasets4',data)
