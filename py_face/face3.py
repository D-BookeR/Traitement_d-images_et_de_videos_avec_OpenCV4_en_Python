import glob
import os
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
CODE_TOUCHE_PAUSE = 32
CODE_TOUCHE_CONTINUE = 32
CAMAPI = cv.CAP_DSHOW
PATH_BASE = 'g:/lib/livreopencv/sourcepython/base/'

try:
    import wx
    mon_appli = wx.App()
except ImportError:
    mon_appli = None

class BaseVisages:
    def __init__(self):
        self.nb_classe = 10
        self.img_classe = []
        self.base_donnees = None
        try:
            self.machine = cv.ml.SVM_load("opencv_ml_svm_faces.xml")
        except (cv.error, SystemError, IOError):
            self.machine = None

        for idx in range(self.nb_classe):
            self.img_classe.append([])
        self.rect_selec = -1
        self.charger_base()

    def gestion_souris(self, event, souris_x, souris_y, _, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.rect_selec = -1
            for index, rect in enumerate(self.liste_rect):
                [x_gauche, y_haut, largeur, hauteur] = rect
                if x_gauche < souris_x < x_gauche + largeur and \
                    y_haut <= souris_y < y_haut + hauteur:
                    self.rect_selec = index
                    break

    def affiche_classe(self):
        for idx, liste_image in enumerate(self.img_classe):
            planche = np.zeros((96 * (1 + len(liste_image) // 4), 4 * 96, 3),
                               np.uint8)
            for idx_image, img in enumerate(liste_image):
                lig = 96 * (idx_image // 4)
                col = 96 * (idx_image % 4)
                planche[lig:lig+96, col:col+96] = cv.resize(img, dsize=(96, 96))
            cv.imshow("classe "+ str(idx), planche)

    def ajout_visage_base(self, img, liste_rect):
        code = 0
        self.liste_rect = liste_rect
        nom_fenetre = "Detection de visage"
        cv.namedWindow(nom_fenetre)
        cv.setMouseCallback(nom_fenetre, self.gestion_souris)
        while code != CODE_TOUCHE_CONTINUE:
            cv.imshow(nom_fenetre, img)
            code = cv.waitKey(10)
            if 48 <= code <= 57 and self.rect_selec != -1:
                idx = code - 48
                rect = liste_rect[self.rect_selec]
                visage = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                nom_fichier = "visage_"+str(idx)+"_"+str(len(self.img_classe[idx]))+"_.png"
                cv.imwrite(PATH_BASE+nom_fichier, visage)
                (self.img_classe[idx]).append(visage)
                self.affiche_classe()
                self.rect_selec = -1
        cv.destroyWindow(nom_fenetre)

    def affiche_erreur_modele(self):
        print("Modele ", self.machine.getDefaultName())
        retval, _ = self.machine.calcError(self.base_donnees, True, None)
        print("Erreur sur les donnees tests : ", retval)
        retval, _ = self.machine.calcError(self.base_donnees, False, None)
        print("Erreur sur les donnees d'entrainement : ", retval)

    def charger_base(self):
        for nom_fichier in glob.glob(PATH_BASE+"visage*.png"):
            visage = cv.imread(nom_fichier)
            idx = nom_fichier.split('_')
            (self.img_classe[int(idx[1])]).append(visage)
        self.affiche_classe()

    def init_svm(self, net):
        self.base_donnee = None
        nb_donnees = 0
        for idx, liste_image in enumerate(self.img_classe):
            nb_donnees = nb_donnees + len(liste_image)
        liste_desc = []
        responses = np.zeros((nb_donnees, 1), np.float32)
        ind_classe = 0
        nb_donnees = 0
        for idx, liste_image in enumerate(self.img_classe):
            for idx_image, img in enumerate(liste_image):
                inp = cv.dnn.blobFromImage(img, 1/255., size=(96, 96), swapRB=True)
                net.setInput(inp)
                val = net.forward()
                liste_desc.append(val.copy().flatten())
                responses[nb_donnees] = ind_classe
                nb_donnees = nb_donnees + 1
            if len(liste_image) != 0:
                ind_classe = ind_classe + 1
        tab_desc = np.array(liste_desc)
        type_variable = np.zeros((tab_desc.shape[1] + 1, 1), np.uint8)
        type_variable.fill(cv.ml.VAR_NUMERICAL)
        type_variable[tab_desc.shape[1], 0] = cv.ml.VAR_CATEGORICAL
        self.base_donnees = cv.ml.TrainData_create(tab_desc, cv.ml.ROW_SAMPLE, responses,
                                                   None, None, None, type_variable)
        self.base_donnees.shuffleTrainTest()
        self.machine = cv.ml.SVM_create()
        self.machine.setKernel(cv.ml.SVM_RBF)
        self.machine.setType(cv.ml.SVM_C_SVC)
        self.machine.setGamma(1)
        self.machine.setC(1)
        self.machine.train(self.base_donnees)
        if self.machine.isTrained():
            self.affiche_erreur_modele()
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return self.machine
        self.machine.save(self.machine.getDefaultName()+"_faces" + ".xml")


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

def zones_visages(val_sorties, taille_image, seuil_confiance, seuil_non_maximum):
    boites, confiance_boites, classe_boites = selection_zones(
        val_sorties[0, 0], seuil_confiance, taille_image)
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
    return facescaffe

if __name__ == '__main__':
#    path = "c:/Users/laurent/Documents/testDNN/openface/"
    path = "f:/testDNN/openface/"
    netFaceCaffe = cv.dnn.readNet(path+"res10_300x300_ssd_iter_140000.caffemodel",
                                  path+"deploy.prototxt")
    seuil_confiance = 0.8
    seuil_non_maximum = 0.1
    net = cv.dnn.readNet(path+"nn4.small2.v1.t7")
    base = BaseVisages()
    fenetre_identification = 'Detection de visage'
    if mon_appli is None:
        nom_video = 'F:/Mes Video/La_Tristitude.mp4'
    else:
        nom_video = wx.FileSelector(
            "Image",
            wildcard="video mp4  (*.mp4)|*.mp4|video avi  (*.avi)|*.avi")
    video = cv.VideoCapture(nom_video)
    code = 0
    while code != CODE_TOUCHE_FIN:
        ret, img = video.read()
        if not ret:
            break

        inp = cv.dnn.blobFromImage(img, 1, size=(300, 300), swapRB=True)
        netFaceCaffe.setInput(inp)
        val_sorties = netFaceCaffe.forward()
        facescaffe = zones_visages(val_sorties, img.shape,
                                   seuil_confiance, seuil_non_maximum)
        visage_image = img.copy()

        if facescaffe is not None:
            for rect in facescaffe:
                cv.rectangle(visage_image,
                             rec=(rect[0], rect[1], rect[2], rect[3]),
                             color=(255, 0, 0))
            if base.machine is not None:
                for rect in facescaffe:
                    inp = cv.dnn.blobFromImage(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]],
                                               1/255., size=(96, 96), swapRB=True)
                    net.setInput(inp)
                    signature = net.forward()
                    ret, ind_classe = base.machine.predict(signature)
                    pos = int(ind_classe[0])
                    cv.putText(visage_image, str(pos), (rect[0], rect[1]+20),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (128, 255, 128), 3)
            cv.imshow(fenetre_identification, visage_image)
            code = cv.waitKey(10)
            if code == CODE_TOUCHE_PAUSE:
                base.ajout_visage_base(visage_image, facescaffe)
            elif code == ord('v'):
                base.init_svm(net)
        else:
            cv.imshow(fenetre_identification, visage_image)
            code = cv.waitKey(10)

cv.destroyAllWindows()
