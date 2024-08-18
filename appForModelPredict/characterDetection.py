import cv2
import numpy as np
def character_detection(image, sizeOfImage):
    if len(image.shape) > 2:
        image = image[:, :, 0]

    maxw = max(image.shape)*2
    newimage = np.ones((maxw, maxw), dtype='uint8')
    size = int(.25*maxw)
    newimage[size:size + image.shape[0], size:size + image.shape[1]] = 255- image
    image = 255 - newimage

    _, image = cv2.threshold(image, 250, 255,cv2.THRESH_BINARY_INV)
    tofind = image

    tofind = cv2.dilate(tofind, np.ones((3, 3)),iterations=20)
    tofind = cv2.erode(tofind, np.ones((3, 3)), iterations=20)

    contour, _ = cv2.findContours(tofind, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    larg_contour = max(contour, key= cv2.contourArea)
    x, y, width, height = cv2.boundingRect(larg_contour)

    ratio = 1 - width/height
    if ratio > 0:
        image = image[y - 5:y+height+5, x - int(height*ratio/2) - 5: x+width + 5 + int(height*ratio/2)]
    else :
        ratio = 1 - height/width
        image = image[y - 5 - int(width*ratio/2):y+height+5 + int(width*ratio/2), x - 5: x+width + 5]

    image = cv2.resize(image, (500, 500))
    image = cv2.blur(image, (9, 9))
    image = cv2.resize(image, sizeOfImage)

    return image