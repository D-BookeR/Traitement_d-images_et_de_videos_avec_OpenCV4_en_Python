import numpy as np
import cv2 as cv

img = cv.imread('g:/lib/livreOpencv/Figure/OCV_Haribo.png')
cv.imshow("Original", img)
cv.waitKey(10)
largeur, hauteur = 350 ,100
b1 = cv.dnn.blobFromImage(img, 1.0, (largeur, hauteur), swapRB=True, crop=False)
imgb = cv.dnn.imagesFromBlob(b1)
cv.imshow("1 swapRB=True crop=False",imgb[0].astype(np.uint8))
cv.waitKey(10)
b1 = cv.dnn.blobFromImage(img, 1.0, (largeur, hauteur), swapRB=False, crop=False)
imgb = cv.dnn.imagesFromBlob(b1)
cv.imshow("2 swapRB=False crop=False",imgb[0].astype(np.uint8))
cv.waitKey(10)
b1 = cv.dnn.blobFromImage(img, 1.0, (largeur, hauteur), swapRB=False, crop=True)
imgb = cv.dnn.imagesFromBlob(b1)
cv.imshow("3 swapRB=False crop=True",imgb[0].astype(np.uint8))
cv.waitKey(10)
b1 = cv.dnn.blobFromImage(img, 1.0, (largeur, hauteur),(20, 0, 0), swapRB=True, crop=True)
imgb = cv.dnn.imagesFromBlob(b1)
cv.imshow("4 swapRB=True crop=True mean",imgb[0].astype(np.uint8))
cv.waitKey(10)
b1 = cv.dnn.blobFromImage(img, 1.0, (largeur, hauteur),(20, 0, 0), swapRB=False, crop=True)
imgb = cv.dnn.imagesFromBlob(b1)
cv.imshow("5 swapRB=False crop=True mean",imgb[0].astype(np.uint8))
s = max([largeur / img.shape[1], hauteur / img.shape[0]])
r = (int(0.5/s * (int(img.shape[1]*s) - largeur)),
            int(0.5/s * (int(img.shape[0]*s) - hauteur)),
            int(largeur/s), int(hauteur / s))
cv.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)
cv.imshow("Original", img)

cv.waitKey(10)
cv.waitKey()
