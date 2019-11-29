from tkinter.filedialog import askopenfilename
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
class Seuillage:
    def __init__(self):
        self.file_name = ""
        self.lut_rnd = []
        self.methode = 0
        self.maj_valeur = True
        self.maj_type = False
        self.maj_aide = True
        self.seuil = 73
        self.k = 0.5
        self.img = []
        self.dst = []
        self.labels = []
        self.aide_txt = []
        self.image_aide = []
        self.stat = []
        self.centroids = []
        self.nom_methode = [
            'Seuillage Manuel', 'Seuillage Otsu',
            'Seuillage Triangle', 'Seuillage local moyenne',
            'Seuillage local gaussienne', 'Seuillage Niblack',
            'Seuillage Niblack SAUVOLA', 'Seuillage Niblack WOLF',
            'Seuillage Niblack NICK'
            ]
    def seuillage_manuel(self):
        _, self.dst = cv.threshold(self.img, type=cv.THRESH_BINARY,
                                   thresh=self.seuil, maxval=255)
    def seuillage_otsu(self):
        _, self.dst = cv.threshold(self.img, type=cv.THRESH_OTSU + cv.THRESH_BINARY,
                                   thresh=self.seuil, maxval=255)
    def seuillage_triangle(self):
        _, self.dst = cv.threshold(self.img, type=cv.THRESH_TRIANGLE + cv.THRESH_BINARY,
                                   thresh=self.seuil, maxval=255)
    def seuillage_local1(self):
        self.dst = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY, 2 * self.seuil + 1, self.k)
    def seuillage_local2(self):
        self.dst = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 2 * self.seuil + 1, self.k)
    def seuillage_niblack(self):
        self.dst = cv.ximgproc.niBlackThreshold(
            self.img, maxValue=255,
            k=self.k, type=cv.THRESH_BINARY,
            blockSize=2 * self.seuil + 1
            )
    def seuillage_niblack_sauvola(self):
        self.dst = cv.ximgproc.niBlackThreshold(
            self.img, maxValue=255,
            k=self.k, type=cv.THRESH_BINARY,
            blockSize=2 * self.seuil + 1,
            binarizationMethod=cv.ximgproc.BINARIZATION_SAUVOLA
            )
    def seuillage_niblack_wolf(self):
        self.dst = cv.ximgproc.niBlackThreshold(
            self.img, maxValue=255,
            k=self.k, type=cv.THRESH_BINARY,
            blockSize=2 * self.seuil + 1,
            binarizationMethod=cv.ximgproc.BINARIZATION_WOLF
            )
    def seuillage_niblack_nick(self):
        self.dst = cv.ximgproc.niBlackThreshold(
            self.img, maxValue=255,
            k=self.k, type=cv.THRESH_BINARY,
            blockSize=2 * self.seuil + 1,
            binarizationMethod=cv.ximgproc.BINARIZATION_NICK
            )

    def maj_methode(self, val_glissiere):
        self.methode = val_glissiere
        self.maj_type = True

    def maj_seuil(self, val_glissiere):
        self.seuil = val_glissiere
        self.maj_valeur = True

    def maj_ratio(self, val_glissiere):
        self.k = (val_glissiere - 100) / 100
        self.maj_valeur = True

    def recherche_composante_connexe(self):
        self.aide_txt = []
        self.maj_aide = True
        nb_composante, self.labels, self.stat, self.centroids = cv.connectedComponentsWithStats(
            self.dst,
            labels=None,
            connectivity=8, ltype=cv.CV_16U)
        negative = cv.bitwise_not(self.dst)
        _, labelsnot, statnot, centroidsnot = cv.connectedComponentsWithStats(
            negative,
            labels=None,
            connectivity=8, ltype=cv.CV_16U)
        self.stat = np.concatenate((self.stat[1:, :], statnot[1:, :]), axis=0)
        self.centroids = np.concatenate((self.centroids[1:, :], centroidsnot[1:, :]), axis=0)
        masque_compo_nulle = self.labels == 0
        labelsnot[masque_compo_nulle] = labelsnot[masque_compo_nulle] + nb_composante - 1
        self.labels = self.labels + labelsnot-1
        return np.uint8(self.labels)
    def aide_attribut(self, event, souris_x, souris_y, flag, param):
        if event == cv.EVENT_LBUTTONDOWN:
            index = self.labels[souris_y, souris_x]
            if index not in self.aide_txt:
                self.aide_txt.append(index)
                self.maj_aide = True
        if event == cv.EVENT_RBUTTONDOWN:
            index = self.labels[souris_y, souris_x]
            if index in self.aide_txt:
                self.aide_txt.remove(index)
                self.maj_aide = True
    def run(self):
        methode_seuillage = {
            0: self.seuillage_manuel,
            1: self.seuillage_otsu,
            2: self.seuillage_triangle,
            3: self.seuillage_local1,
            4: self.seuillage_local2,
            5: self.seuillage_niblack,
            6: self.seuillage_niblack_sauvola,
            7: self.seuillage_niblack_wolf,
            8: self.seuillage_niblack_nick
            }
        self.file_name = askopenfilename()
        self.img = cv.imread(self.file_name, cv.IMREAD_GRAYSCALE)
        if self.img is None:
            print('Le fichier image ne peut etre lu- Verifier le chemin')
            return
        methode_seuillage[0]()
        cv.namedWindow(self.file_name)
        ajouter_glissiere(
            "Methode", self.file_name,
            0, len(methode_seuillage) - 1,
            self.methode, self.maj_methode)
        ajouter_glissiere("Seuil", self.file_name, 0, 255, self.seuil, self.maj_seuil)
        lut_rnd = np.random.randint(0, 256, size=(256, 1, 3), dtype=np.uint8)
        cv.setMouseCallback(self.file_name, self.aide_attribut)
        self.image_aide = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        while True:
            if self.maj_aide:
                self.image_aide = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
                for i in self.aide_txt:
                    mask = self.labels == i
                    moyenne = cv.mean(self.img, mask.astype(np.uint8))
                    text = '{:.4}'.format(moyenne[0]) + ': ' + str(self.stat[i][4])
                    pos = (int(self.centroids[i][0]), int(self.centroids[i][1]))
                    cv.putText(
                        self.image_aide, text,
                        pos, cv.FONT_HERSHEY_SIMPLEX,
                        0.5, tuple(lut_rnd[i % 256, 0, :].tolist()),
                        2
                        )
                self.maj_aide = False
            cv.imshow(self.file_name, self.image_aide)
            if self.maj_valeur:
                methode_seuillage[self.methode]()
                cv.imshow("threh image", self.dst)
                labels = self.recherche_composante_connexe()
                dst = cv.applyColorMap(labels, lut_rnd)
                cv.imshow("label", dst)
                self.maj_valeur = False
            if self.maj_type:
                print("Methode:", self.nom_methode[self.methode])
                cv.destroyWindow(self.file_name)
                cv.namedWindow(self.file_name)
                cv.setMouseCallback(self.file_name, self.aide_attribut)
                ajouter_glissiere(
                    "Method", self.file_name,
                    0, len(methode_seuillage) - 1,
                    self.methode, self.maj_methode
                    )
                if self.methode == 0:
                    ajouter_glissiere(
                        "thresh", self.file_name,
                        0, 255, self.seuil, self.maj_seuil
                        )
                elif self.methode >= 3:
                    ajouter_glissiere(
                        "Bloc", self.file_name,
                        1, 255, self.seuil,
                        self.maj_seuil
                        )
                    ajouter_glissiere(
                        "Ratio", self.file_name,
                        1, 200, 150, self.maj_seuil
                        )
                self.maj_type = False
                self.maj_valeur = True
            code_touche = cv.waitKey(10)
            if code_touche == 27:
                break
        cv.destroyAllWindows()
if __name__ == '__main__':
    Seuillage().run()
