import numpy as np
import cv2 as cv

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
class Filtrage:
    def __init__(self):
        self.type_gradient = 0
        self.nom_fichier = ""
        self.maj_type_gradient = False
        self.maj_param_gradient = False
        self.maj_canny = True
        self.taille = 0
        self.alpha = 50
        self.omega = 50
        self.img = []
        self.dx_img = []
        self.dy_img = []
        self.module = []
        self.fct_gradient = {
            0: self.filtre_sobel,
            1: self.filtre_scharr,
            2: self.filtre_deriche,
            }
    def glissiere_sobel(self):
        cv.destroyWindow(self.nom_fichier)
        cv.namedWindow(self.nom_fichier)
        ajouter_glissiere("Filtre", self.nom_fichier, 0, 2,
                          self.type_gradient, self.glissiere_filtre)
        ajouter_glissiere("Taille", self.nom_fichier, 0, 15,
                          self.taille, self.glissiere_taille)
    def glissiere_scharr(self):
        cv.destroyWindow(self.nom_fichier)
        cv.namedWindow(self.nom_fichier)
        ajouter_glissiere("Filtre", self.nom_fichier, 0, 2,
                          self.type_gradient, self.glissiere_filtre)
    def glissiere_deriche(self):
        cv.destroyWindow(self.nom_fichier)
        cv.namedWindow(self.nom_fichier)
        ajouter_glissiere("Filtre", self.nom_fichier, 0, 2,
                          self.type_gradient, self.glissiere_filtre)
        ajouter_glissiere("alpha", self.nom_fichier, 1, 400,
                          self.alpha, self.glissiere_alpha)
        ajouter_glissiere("omega", self.nom_fichier, 1, 1000,
                          self.omega, self.glissiere_omega)
    def glissiere_omega(self, val_glissiere):
        self.omega = val_glissiere
        self.fct_gradient[self.type_gradient]()
        self.module_gradient()
        cv.imshow("module", self.module)
        self.maj_canny = True
    def glissiere_alpha(self, val_glissiere):
        self.alpha = val_glissiere
        self.fct_gradient[self.type_gradient]()
        self.module_gradient()
        cv.imshow("module", self.module)
        self.maj_canny = True
    def glissiere_canny(self, _):
        self.maj_canny = True
    def glissiere_filtre(self, val_glissiere):
        self.type_gradient = val_glissiere
        self.maj_type_gradient = True
    def glissiere_taille(self, val_glissiere):
        self.taille = val_glissiere
        self.fct_gradient[self.type_gradient]()
        self.module_gradient()
        cv.imshow("module", self.module)
        self.maj_canny = True
    def filtre_sobel(self):
        self.dx_img = cv.Sobel(self.img, ddepth=cv.CV_32F,
                               dx=1, dy=0, ksize=2 * self.taille + 1)
        self.dy_img = cv.Sobel(self.img, ddepth=cv.CV_32F,
                               dx=0, dy=1, ksize=2 * self.taille + 1)
    def filtre_scharr(self):
        self.dx_img = cv.Scharr(self.img, ddepth=cv.CV_32F, dx=1, dy=0)
        self.dy_img = cv.Scharr(self.img, ddepth=cv.CV_32F, dx=0, dy=1)
    def filtre_deriche(self):
        self.dx_img = cv.ximgproc.GradientDericheX(self.img,
                                                   self.alpha / 100,
                                                   self.omega / 1000)
        self.dy_img = cv.ximgproc.GradientDericheY(self.img,
                                                   self.alpha / 100,
                                                   self.omega / 1000)
    def module_gradient(self):
        dx2 = self.dx_img * self.dx_img
        dy2 = self.dy_img * self.dy_img
        self.module = np.sqrt(dx2 + dy2)
        k = 100 / self.module.max()
        self.dx_img = self.dx_img * k
        self.dy_img = self.dy_img * k
        self.module = cv.normalize(src=self.module, dst=None,
                                   norm_type=cv.NORM_MINMAX,
                                   alpha=255, dtype=cv.CV_8U)

    def run(self):
        filtre_glissiere = {
            0: self.glissiere_sobel,
            1: self.glissiere_scharr,
            2: self.glissiere_deriche,
            }
        self.nom_fichier = '../Figure/ocv_haribo.png'
        self.img = cv.imread(self.nom_fichier, cv.IMREAD_GRAYSCALE)
        if self.img is None:
            print('Le fichier image ne peut etre lu')
            return
        filtre_glissiere[0]()
        self.fct_gradient[self.type_gradient]()
        cv.namedWindow("Canny edges")
        ajouter_glissiere("seuilHaut", "Canny edges",
                          0, 100, 50, self.glissiere_canny)
        ajouter_glissiere("seuilBas", "Canny edges",
                          0, 100, 10, self.glissiere_canny)
        lut_rnd = np.random.randint(0, 256, size=(256, 1, 3),
                                    dtype=np.uint8)
        while True:
            cv.imshow(self.nom_fichier, self.img)
            if self.maj_type_gradient:
                filtre_glissiere[self.type_gradient]()
                self.fct_gradient[self.type_gradient]()
                self.module_gradient()
                cv.imshow("module", self.module)
                self.maj_type_gradient = False
                self.maj_canny = True
            if self.maj_canny:
                seuil_haut = cv.getTrackbarPos("seuilHaut", "Canny edges")
                seuil_bas = cv.getTrackbarPos("seuilBas", "Canny edges")
                front = cv.Canny(self.dx_img.astype(np.short),
                                 self.dy_img.astype(np.short),
                                 seuil_bas, seuil_haut)
                contours, _ = cv.findContours(front, cv.RETR_LIST,
                                              cv.CHAIN_APPROX_NONE)
                print("Nombre de contours : ", len(contours))
                labels = np.zeros((front.shape[0], front.shape[1], 3),
                                  dtype=np.uint8)
                cv.imshow("Canny edges", front)
                for idx, _ in enumerate(contours):
                    if len(contours[idx]) < 1000 and len(contours[idx]) > 50:
                        cv.drawContours(labels, contours, idx,
                                        tuple(lut_rnd[idx%256, 0, :].tolist()),
                                        cv.FILLED)
                cv.drawContours(labels, contours, -1, (0, 255, 0), 1)
                cv.imshow("region", labels)
                self.maj_canny = False
            code_touche_clavier = cv.waitKey(10)
            if code_touche_clavier == 27:
                break
        cv.destroyAllWindows()
if __name__ == '__main__':
    Filtrage().run()
