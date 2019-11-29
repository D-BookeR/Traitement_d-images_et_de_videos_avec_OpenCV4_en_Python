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

def choisir_neural_modele():
    if mon_appli:
        nom_modele = wx.FileSelector("Choississez un modèle .t7", wildcard='*.t7')
    else:
        path = "./"
        nom_modele = path + "modele.t7"
    return nom_modele


class GlissiereEffet:
    def __init__(self, nom_fenetre, nom_glissieres, val_glissiere, fct, modele=None):
        self.nom_fenetre = nom_fenetre
        cv.namedWindow(nom_fenetre)
        self.fct = fct
        self.modele = modele
        self.image_vide = None
        self.nom_glissiere = ['Actif']
        self.echelle = [1]
        self.actif = False
        ajouter_glissiere('Actif', self.nom_fenetre, 0, 1, 0, self.activation)
        for nom, val in zip(nom_glissieres, val_glissiere):
            self.nom_glissiere.append(nom)
            if val[3] == 0:
                k = 1
            else:
                k = val[3]
            ajouter_glissiere(nom, self.nom_fenetre, val[0], val[1], int(val[2] * k), None)
            self.echelle.append(val[3])
    def activation(self, valeur):
        self.actif = int(cv.getTrackbarPos('Actif', self.nom_fenetre)) == 1
    def lecture_glissiere(self):
        val_retour = []
        for nom, val in zip(self.nom_glissiere, self.echelle):
            if val == 0:
                val_retour.append(int(cv.getTrackbarPos(nom, self.nom_fenetre)))
            else:
                val_retour.append(cv.getTrackbarPos(nom, self.nom_fenetre) / val)
        return val_retour
    def appliquer(self, src):
        parametre = self.lecture_glissiere()
        if parametre[0] == 0:
            self.actif = False
            if self.image_vide is None:
                self.image_vide = np.zeros(img.shape, np.uint8)
                if self.modele is None:
                    cv.putText(self.image_vide, "Effet inactif",
                               (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1, (64, 0, 128), 1)
                else:
                    cv.putText(self.image_vide, "Modele inactif",
                               (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1, (64, 0, 128), 1)
            return self.image_vide
        self.actif = True
        if self.fct is not None:
            if self.fct == cv.edgePreservingFilter:
                dst = self.fct(src, flags=cv.NORMCONV_FILTER,
                               sigma_s=parametre[1],
                               sigma_r=parametre[2])
            elif self.fct == cv.pencilSketch:
                dst, _ = self.fct(src,
                                  sigma_s=parametre[1],
                                  sigma_r=parametre[2],
                                  shade_factor=parametre[3])
            elif self.fct == cv.xphoto.oilPainting:
                dst = self.fct(src,
                               size=parametre[1],
                               dynRatio=parametre[2])
            else:
                dst = self.fct(src,
                               sigma_s=parametre[1],
                               sigma_r=parametre[2])
            return dst
        elif self.modele is not None:
            inp = cv.dnn.blobFromImage(src, 1.0, 
                                       (src.shape[1], src.shape[0]),
                                       (103.939, 116.779, 123.68),
                                       swapRB=False, crop=False)
            self.modele.setInput(inp)
            out = self.modele.forward()
            out = out.reshape(3, out.shape[2], out.shape[3])
            out[0] += 103.939
            out[1] += 116.779
            out[2] += 123.68
            out = np.clip(out.transpose(1, 2, 0), 0, 255)
            return out.astype(np.uint8)
        return None

if __name__ == '__main__':

    video = cv.VideoCapture(CAMAPI)
    code = 0
    ret, img = video.read()
    code = ord('o')
    nom_fenetre = []
    liste_effets = []
    liste_modeles = []
    nom_modele = choisir_neural_modele()
    while len(nom_modele) != 0:
        net = cv.dnn.readNetFromTorch(nom_modele)
        nom_fichier = nom_modele.split('\\')[-1]
        liste_effets.append(GlissiereEffet(
            nom_fichier + "_" + 'detail',
            ['sigma_s', 'sigma_r'],
            [[0, 200, 60, 1],
            [0, 100, 0.4, 100]],
            cv.detailEnhance))
        liste_effets.append(GlissiereEffet(
            nom_fichier + "_" + 'front', 
            ['sigma_s', 'sigma_r'],
            [[0, 200, 60, 1],
            [0, 100, 0.4, 100]],
            cv.edgePreservingFilter))
        liste_effets.append(GlissiereEffet(
            nom_fichier + "_" + 'schema',
            ['sigma_s', 'sigma_r', 'ombre'],
            [[0, 200, 60, 1],
             [0, 100, 0.4, 100],
             [0, 10, 7, 100]],
             cv.pencilSketch))
        liste_effets.append(GlissiereEffet(
            nom_fichier + "_" + 'Style', 
            ['sigma_s', 'sigma_r'],
            [[0, 200, 60, 1],
             [0, 100, 0.45, 100]],
             cv.stylization))
        liste_effets.append(GlissiereEffet(
            nom_fichier + "_" + 'Huile',
            ['taille', 'dyn'],
            [[3, 20, 3, 0],
             [1, 100, 16, 0]],
             cv.xphoto.oilPainting))
        liste_modeles.append(
            GlissiereEffet(nom_fichier,
                            [], [], None, net))
        nom_modele = choisir_neural_modele()

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
            for modele in liste_modeles:
                if modele.actif:
                    dst = modele.appliquer(img)
                    cv.imshow(modele.nom_fenetre, dst)
                    cv.imwrite(modele.nom_fenetre+".png",dst)
                    for effet in liste_effets:
                        if effet.actif and modele.nom_fenetre in modele.nom_fenetre:
                            dst_net = effet.appliquer(dst)
                            cv.imwrite(effet.nom_fenetre+".png",dst_net)
                            cv.imshow(effet.nom_fenetre, dst_net)
            if video.isOpened():
                code = cv.waitKey(10)
            else:
                code = cv.waitKey()

    cv.destroyAllWindows()
    exit()
