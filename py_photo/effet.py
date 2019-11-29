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

class GlissiereEffet:
    def __init__(self, nom_fenetre, nom_glissieres, val_glissiere, fct):
        self.nom_fenetre = nom_fenetre
        cv.namedWindow(nom_fenetre)
        self.fct = fct
        self.image_vide = None
        self.nom_glissiere = ['Actif']
        self.echelle = [1]
        ajouter_glissiere('Actif', self.nom_fenetre, 0, 1, 0, None)
        for nom, val in zip(nom_glissieres, val_glissiere):
            self.nom_glissiere.append(nom)
            if val[3] == 0:
                k = 1
            else:
                k = val[3]
            ajouter_glissiere(nom, self.nom_fenetre, val[0], val[1], int(val[2] * k), None)
            self.echelle.append(val[3])

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
            if self.image_vide is None:
                self.image_vide = np.zeros(img.shape, np.uint8)
                cv.putText(self.image_vide, " Effet inactif",
                           (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                           1, (64, 0, 128), 1)
            return self.image_vide
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


if __name__ == '__main__':

    video = cv.VideoCapture(CAMAPI)
    code = 0
    ret, img = video.read()
    code = ord('o')
    nom_fenetre = ['detail', 'front', 'schema', 'stylisation', 'huile']
    liste_effet = []
    liste_effet.append(
        GlissiereEffet(nom_fenetre[0], ['sigma_s', 'sigma_r'],
                       [[0, 200, 60, 1],
                        [0, 100, 0.4, 100]],
                        cv.detailEnhance))
    liste_effet.append(
        GlissiereEffet(nom_fenetre[1], ['sigma_s', 'sigma_r'],
                       [[0, 200, 60, 1],
                        [0, 100, 0.4, 100]],
                        cv.edgePreservingFilter))
    liste_effet.append(
        GlissiereEffet(nom_fenetre[2],
                       ['sigma_s', 'sigma_r', 'ombre'],
                       [[0, 200, 60, 1],
                        [0, 100, 0.4, 100],
                        [0, 10, 7, 100]],
                        cv.pencilSketch))
    liste_effet.append(
        GlissiereEffet(nom_fenetre[3],
                       ['sigma_s', 'sigma_r'],
                       [[0, 200, 60, 1],
                        [0, 100, 0.45, 100]],
                        cv.stylization))
    liste_effet.append(
        GlissiereEffet(nom_fenetre[4],
                       ['taille', 'dyn'],
                       [[3, 20, 3, 0],
                        [1, 100, 16, 0]],
                        cv.xphoto.oilPainting))

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
            for effet in liste_effet:
                dst = effet.appliquer(img)
                cv.imshow(effet.nom_fenetre, dst)
                if video.isOpened() is False:
                    cv.imwrite("opencv-book-" + effet.nom_fenetre + ".png", dst)

            if video.isOpened():
                code = cv.waitKey(10)
            else:
                code = cv.waitKey()

    cv.destroyAllWindows()
    exit()
