import cv2 as cv

NBLIGNE = 8
NBCOLONNE = 5
DICO = 14

dico = cv.aruco.Dictionary_get(DICO)
board= cv.aruco.CharucoBoard_create(NBLIGNE, NBCOLONNE,  1, 0.5, dico)
img = board.draw((1000,1000), 50, 50)
cv.imshow('ChaRuCo', img)
cv.waitKey()
cv.imwrite('charuco.png', img)
