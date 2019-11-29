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

def selection_zones(val_sorties, seuil_confiance, taille_image):
    val_confiances = []
    zones = []
    ind_classes = []
    hauteur_image, largeur_image, _ = taille_image
    for detection in val_sorties:
        confiance = detection[2]
        indice_classe = int(detection[1])
        if confiance > seuil_confiance and indice_classe == 1:
            pos_gauche = int(detection[3] * largeur_image)
            pos_haut = int(detection[4] * hauteur_image)
            pos_droite = int(detection[5] * largeur_image)
            pos_bas = int(detection[6] * hauteur_image)
            largeur = pos_droite - pos_gauche
            hauteur = pos_bas - pos_haut
            if largeur > 0 and hauteur > 0:
                zones.append([pos_gauche, pos_haut, largeur, hauteur])
                ind_classes.append(indice_classe)
                val_confiances.append(float(confiance))
    return zones, val_confiances, ind_classes

if __name__ == '__main__':
    path = "f:/testDNN/openface/"
    netFaceCaffe = cv.dnn.readNet(path+"res10_300x300_ssd_iter_140000.caffemodel",
                                  path+"deploy.prototxt")
    nom_sorties = netFaceCaffe.getUnconnectedOutLayersNames()
    seuil_confiance = 0.8
    seuil_non_maximum = 0.1

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

        inp = cv.dnn.blobFromImage(img, 1, size=(300, 300), swapRB=True)
        netFaceCaffe.setInput(inp)
        val_sorties = netFaceCaffe.forward()
        boites, confiance_boites, classe_boites = selection_zones(
            val_sorties[0, 0], seuil_confiance, img.shape)
        indices_retenus = cv.dnn.NMSBoxes(
            boites, confiance_boites,
            seuil_confiance, seuil_non_maximum
            )

        facescaffe = None
        for indice in indices_retenus:
            pos_gauche = boites[indice[0]][0]
            pos_haut = boites[indice[0]][1]
            if pos_gauche < 0:
                pos_gauche = 0
            if pos_haut < 0:
                pos_haut = 0
            largeur = boites[indice[0]][2]
            hauteur = boites[indice[0]][3]
            indice_classe = classe_boites[indice[0]]
            if facescaffe is not None:
                facescaffe = np.vstack((facescaffe, [pos_gauche, pos_haut, largeur, hauteur]))
            else:
                facescaffe = np.array([(pos_gauche, pos_haut, largeur, hauteur)], dtype=np.int32)

        if facescaffe is not None:
            for rect in facescaffe:
                cv.rectangle(img, rec=(rect[0], rect[1], rect[2], rect[3]),
                             color=(255, 0, 0))
        cv.imshow(nom_fenetre, img)
    cv.destroyAllWindows()
