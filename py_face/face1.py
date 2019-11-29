import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
CAMAPI = cv.CAP_DSHOW

try:
    import wx
    mon_appli = wx.App()
except ImportError:
    mon_appli = None

cascade = cv.CascadeClassifier("F:/lib/opencv_extra/testdata/cv/face/lbpcascade_frontalface_improved.xml")
nom_fenetre = 'Detection de visage'
if mon_appli is None:
    nom_video = 'F:/Mes Video/La_Tristitude.mp4'
else:
    nom_video = wx.FileSelector(
        "Image",
        wildcard="video mp4  (*.mp4)|*.mp4|video avi  (*.avi)|*.avi")

video = cv.VideoCapture(nom_video)
code = ord('o')
while code != CODE_TOUCHE_FIN:
    if video.isOpened():
        ret, img = video.read()
    else:
        if mon_appli is None:
            nom_image = 'mon_image.jpg'
        if code == ord('o'):
            nom_image = wx.FileSelector(
                "Image",
                wildcard="image jpeg  (*.jpg)|*.jpg|image tiff  (*.tif)|*.tif")
        img = cv.imread(nom_image)
    code = cv.waitKey(10)
    if img is not None:
        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            for rect in faces:
                cv.rectangle(img, rec=(rect[0], rect[1], rect[2], rect[3]), color=(0, 255, 0))
            cv.imshow("Detection de visage Cascade", img)
        else:
            cv.imshow("Detection de visage Cascade", img)
cv.destroyAllWindows()

