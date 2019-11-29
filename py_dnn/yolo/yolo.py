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
                          val_defaut,
                          max_glissiere - min_glissiere + 1,
                          fct_glissiere)
    else:
        cv.createTrackbar(nom_glissiere,
                          nom_fenetre,
                          val_defaut,
                          max_glissiere - min_glissiere + 1,
                          lambda *args: None)
    cv.setTrackbarMin(nom_glissiere,
                      nom_fenetre,
                      min_glissiere)
    cv.setTrackbarMax(nom_glissiere,
                      nom_fenetre,
                      max_glissiere)
    cv.setTrackbarPos(nom_glissiere,
                      nom_fenetre,
                      val_defaut)

def choisir_yolo_modele():
    if mon_appli:
        nom_modele = wx.FileSelector(
            "Fichier des poids",
            wildcard="poids du modèle (*.weights)|*.weights")
        nom_proto = wx.FileSelector(
            "Fichier de configuration",
            wildcard="configuration du modèle (*.cfg)|*.cfg")
        nom_classe = wx.FileSelector(
            "Fichier des classes",
            default_filename="coco.names",
            wildcard="classes du modèle (*.names)|*.names")
    else:
        path = "f:/testDNN/objectdetection/yolov3-spp/"
        nom_modele = path + "yolov3-spp.weights"
        nom_proto = path + "yolov3-spp.cfg"
        nom_classe = path + "coco.names"
    return nom_modele, nom_proto, nom_classe

def selection_zones(val_sorties, seuil_confiance, r_blob):
    val_confiances = []
    zones = []
    ind_classes = []
    for val in val_sorties:
        for att_object in val:
            if att_object[4] > seuil_confiance:
                indice_classe = np.argmax(att_object[5:])
                proba = att_object[5+indice_classe]
                x = int(att_object[0] * r_blob[2]) + r_blob[0]
                y = int(att_object[1] * r_blob[3]) + r_blob[1]
                largeur = int(att_object[2] * r_blob[2])
                hauteur = int(att_object[3] * r_blob[3])
                pos_gauche = int(x - largeur / 2)
                pos_haut = int(y - hauteur / 2)
                zones.append([pos_gauche, pos_haut, largeur, hauteur])
                ind_classes.append(indice_classe)
                val_confiances.append(float(proba))
    return zones, val_confiances, ind_classes

if __name__ == '__main__':
    nom_modele, nom_proto, nom_classe = choisir_yolo_modele()
    yolo = cv.dnn.readNet(nom_modele, nom_proto)
    try:
        with open(nom_classe, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    except:
        classes = None

    nom_fenetre = "Yolo"
    cv.namedWindow(nom_fenetre)
    seuil_confiance = 0.8
    seuil_non_maximum = 0.4
    ajouter_glissiere("SeuilConfiance", nom_fenetre, 0, 100,
                      int(seuil_confiance*100), None)
    ajouter_glissiere("Rogner", nom_fenetre, 0, 1,
                      0, None)
    ajouter_glissiere("SeuilBoite", nom_fenetre, 0, 100,
                      int(seuil_non_maximum*100), None)
    video = cv.VideoCapture(CAMAPI)
    video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    ret, img = video.read()
    code = ord('o')
    nom_sorties = yolo.getUnconnectedOutLayersNames()

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
        if img is not None:
            seuil_confiance = cv.getTrackbarPos("SeuilConfiance", nom_fenetre) / 100
            seuil_non_maximum = cv.getTrackbarPos("SeuilBoite", nom_fenetre) / 100
            rogner = cv.getTrackbarPos("Rogner", nom_fenetre) == 1
            largeur_blob, hauteur_blob = 608, 608
            blob = cv.dnn.blobFromImage(img,
                                        1.0 / 255.0,
                                        (largeur_blob, hauteur_blob),
                                        swapRB=True,
                                        crop=rogner)
            if rogner:
                s = max([largeur_blob / img.shape[1], hauteur_blob / img.shape[0]])
                r_blob = (int(0.5 / s * (int(img.shape[1] * s) - largeur_blob)),
                          int(0.5 / s * (int(img.shape[0] * s) - hauteur_blob)),
                          int(largeur_blob / s),
                          int(hauteur_blob / s))
            else:
                r_blob = (0, 0, img.shape[1], img.shape[0])
            yolo.setInput(blob)
            val_sorties = yolo.forward(nom_sorties)
            boites, confiance_boites, classe_boites = selection_zones(
                val_sorties, seuil_confiance, r_blob)
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
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv.rectangle(img,
                         (r_blob[0], r_blob[1]),
                         (r_blob[0] + r_blob[2], r_blob[1] + r_blob[3]), (0, 0, 255), 2)
            cv.imshow(nom_fenetre, img)
            if video.isOpened():
                code = cv.waitKey(10)
            else:
                code = cv.waitKey()

    cv.destroyAllWindows()
 #   yolo.dumpToFile("yolo.dot")
