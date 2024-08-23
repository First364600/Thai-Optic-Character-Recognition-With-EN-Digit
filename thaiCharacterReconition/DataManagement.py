import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# url = 'AIFORTHAI-ThaiOCRCorpus\ThaiOCR\ThaiOCR-TrainigSet\Thai\Thai - Copy/161/161 (1).bmp'
x, y = [], []
for i in range(161, 207):
    for j in range(1, 3085) :
        url = 'AIFORTHAI-ThaiOCRCorpus/ThaiOCR/ThaiOCR-TrainigSet/Thai/Thai - Copy/' + str(i) + '/' + str(i) + ' (' + str(j) + ').bmp'
        image = Image.open(url)
        imageRe = image.resize((60, 60))
        image = np.asarray(imageRe)
        image = np.where(image == True, 0, 255)
        image = image.astype(np.float64)
        plt.imshow(image, cmap="gray")
        # image = np.stack((image,) *3, axis=-1)
        # cc = cv2.imread('thaiCharacterReconition/4char.png', 0)
        # print(image.shape)
        
        # image = cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY)
        ret, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=4)
        image = cv2.blur(image, (3, 3))
        # blur = cv2.blur(imageRe, (10, 10))
        # print(image)
        imageArray = np.asarray(image)
        plt.show()
        x.append(imageArray)
        y.append(i-161)
    print(i)

data = {
    'image' : np.asarray(x),
    'answer' : np.asarray(y)
}
np.save('thaiDatasets2', data)
print((np.asarray(x)).shape)

plt.imshow(x[10000])
plt.title(y[10000])
plt.show()