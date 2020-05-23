import time
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27

liste_webcam = []
index_camera_ouvertes  = 0
while True:
    webcam = cv.VideoCapture(index_camera_ouvertes)
    if webcam.isOpened():
        liste_webcam.append(webcam)
    else:
        break
    index_camera_ouvertes  += 1
if not liste_webcam:
    print("Aucune camera reliee")
    exit()
code_touche_clavier  = 0
nb_prises = 0
tps_ini = time.clock()
while True:
    for index, cam in enumerate(liste_webcam):
        ret, img = cam.read()
        if ret:
            cv.imshow('webcam b' + str(idx), img)
        else:
            print(" image ne peut être lue")
    nb_prises += 1
    if nb_prises == 100:
        nb_images_par_seconde = nb_prises / (time.clock() - tps_ini)
        print("Pour chaque camera : ",  nb_images_par_seconde, " Images par seconde")
        tps_ini = time.clock()
        nb_prises = 0
    code_touche_clavier  = cv.waitKey(20)
    if code_touche_clavier  == CODE_TOUCHE_ECHAP:
        break
for cam in liste_webcam:
    cam.release()
cv.destroyAllWindows()
