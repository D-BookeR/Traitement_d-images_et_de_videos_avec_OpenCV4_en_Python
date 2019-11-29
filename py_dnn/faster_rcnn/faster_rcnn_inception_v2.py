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

def ajouter_glissiere(nom_glissiere, nom_fenetre,
                      min_glissiere, max_glissiere,
                      val_defaut, fct_glissiere=None):
    if fct_glissiere:
        cv.createTrackbar(nom_glissiere, nom_fenetre,
                          val_defaut, max_glissiere - min_glissiere + 1,
                          fct_glissiere)
    else:
        cv.createTrackbar(nom_glissiere,
                          nom_fenetre,
                          val_defaut,
                          max_glissiere - min_glissiere + 1,
                          lambda *args: None)
    cv.setTrackbarMin(nom_glissiere, nom_fenetre, min_glissiere)
    cv.setTrackbarMax(nom_glissiere, nom_fenetre, max_glissiere)
    cv.setTrackbarPos(nom_glissiere, nom_fenetre, val_defaut)

def choisir_tf_modele():
    if mon_appli:
        nom_modele = wx.FileSelector("Fichier des poids", wildcard="poids du modèle (*.pb)|*.pb")
        nom_proto = wx.FileSelector("Fichier de configuration",
                                    wildcard="configuration du modèle (*.pbtxt)|*.pbtxt")
        nom_classe = wx.FileSelector("Fichier des classes",
                                     default_filename="object_detection_classes_coco.txt",
                                     wildcard="classes du modèle (*.txt)|*.txt")
    else:
        path = "f:/testDNN/objectdetection/faster_rcnn_inception_v2_coco_2018_01_28/"
        nom_modele = path + "frozen_inference_graph.pb"
        nom_proto = path + "frozen.pbtxt"
        nom_classe = path + "object_detection_classes_coco.txt"
    return nom_modele, nom_proto, nom_classe

def selection_zones(val_sorties, seuil_confiance):
    val_confiances = []
    zones = []
    ind_classes = []
    for detection in val_sorties[0, 0]:
        confiance = detection[2]
        indice_classe = int(detection[1])
        if confiance > seuil_confiance:
            pos_gauche = int(detection[3] * largeur_image)
            pos_haut = int(detection[4] * hauteur_image)
            pos_droite = int(detection[5] * largeur_image)
            pos_bas = int(detection[6] * hauteur_image)
            largeur = pos_droite - pos_gauche + 1
            hauteur = pos_bas - pos_haut + 1
            zones.append([pos_gauche, pos_haut, largeur, hauteur])
            ind_classes.append(indice_classe)
            val_confiances.append(float(confiance))
    return zones, val_confiances, ind_classes

if __name__ == '__main__':
    nom_modele, nom_proto, nom_classe = choisir_tf_modele()
    fast_rcnn = cv.dnn.readNet(nom_modele, nom_proto)
    if fast_rcnn.empty():
        print("Le réseau est vide!")
        exit()

    try:
        with open(nom_classe, 'rt') as f:
            classes = f.read().split('\n')
    except:
        classes = None

    nom_fenetre = "Faster_rcnn_inception"
    cv.namedWindow(nom_fenetre)
    seuil_confiance = 0.8
    seuil_non_maximum = 0.4
    ajouter_glissiere("SeuilConfiance", nom_fenetre, 0, 100,
                      int(seuil_confiance*100), None)
    ajouter_glissiere("SeuilBoite", nom_fenetre, 0, 100,
                      int(seuil_non_maximum*100), None)
    video = cv.VideoCapture(CAMAPI)
    video.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    code = 0
    ret, img = video.read()
    if img is not None:
        (hauteur_image, largeur_image, _) = img.shape
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
            (hauteur_image, largeur_image, _) = img.shape
        if img is not None:
            seuil_confiance = cv.getTrackbarPos("SeuilConfiance", nom_fenetre) / 100
            seuil_non_maximum = cv.getTrackbarPos("SeuilBoite", nom_fenetre) / 100
            blob = cv.dnn.blobFromImage(img)
            fast_rcnn.setInput(blob)
            val_sorties = fast_rcnn.forward()
            boites, confiance_boites, classe_boites = selection_zones(
                val_sorties, seuil_confiance)
            indices_retenus = cv.dnn.NMSBoxes(
                boites, confiance_boites,
                seuil_confiance, seuil_non_maximum
                )
            for indice in indices_retenus:
                pos_gauche = boites[indice[0]][0]
                pos_haut = boites[indice[0]][1]
                largeur = boites[indice[0]][2]
                hauteur = boites[indice[0]][3]
                indice_classe = classe_boites[indice[0]]
                cv.rectangle(img, (pos_gauche, pos_haut),
                             (pos_gauche + largeur, pos_haut + hauteur), (0, 255, 0), 2)
                cv.putText(img, classes[indice_classe],
                           (pos_gauche + 5, pos_haut + 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (64, 0, 128), 1)
            cv.imshow(nom_fenetre, img)
            if video.isOpened():
                code = cv.waitKey(10)
            else:
                code = cv.waitKey()

    cv.destroyAllWindows()
