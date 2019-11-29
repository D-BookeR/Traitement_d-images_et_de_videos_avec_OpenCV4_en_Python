import wx
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27

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
class ClassifKMoyennes:
    def __init__(self):
        self.nom_fichier = None
        self.img = None
        self.img_gris = None
        self.couleur = 0
        self.nb_classes = 3
        self.essai = 5
        self.iteration = 10
        self.nuage_couleur = []
        self.nuage_gris = []
        self.meilleur_labels = None
        lut_rnd = 256 * np.random.rand(256, 1, 3)
        self.lut_rnd = lut_rnd.astype(np.uint8)

    def classification(self):
        if self.img_gris is None:
            self.img_gris = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            self.nuage_gris = self.img_gris.astype(np.float32, copy=True)
            self.nuage_gris.resize((self.img.shape[0] * self.img.shape[1], 1))
            self.nuage_couleur = self.img.astype(np.float32, copy=True)
            self.nuage_couleur.resize((self.img.shape[0] * self.img.shape[1], 3))

        critere_fin = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, self.iteration, 1)
        if self.couleur == 1:
            data = self.nuage_couleur
        else:
            data = self.nuage_gris

        if self.meilleur_labels is None:
            condition = cv.KMEANS_RANDOM_CENTERS
        else:
            condition = cv.KMEANS_USE_INITIAL_LABELS

        _, self.meilleur_labels, centres = cv.kmeans(
            data,
            self.nb_classes, self.meilleur_labels,
            criteria=critere_fin, attempts=self.essai,
            flags=condition)
        meilleur_labels = self.meilleur_labels.astype(np.uint8)
        meilleur_labels.resize((self.img.shape[0], self.img.shape[1]))
        if self.couleur == 1:
            centres = centres.astype(np.uint8)
            centres = np.vstack((centres, np.zeros((256 - self.nb_classes, 3), np.uint8)))
            centres.resize((256, 1, 3))
            meilleur_labels = cv.applyColorMap(meilleur_labels, centres)
        else:
            meilleur_labels = cv.applyColorMap(meilleur_labels, self.lut_rnd)
        cv.imshow("Labels", meilleur_labels)

    def efface_fenetre(self, nom_fenetre):
        ecran_attente = np.zeros(self.img.shape, np.uint8)
        cv.putText(ecran_attente, "Classification en cours!",
                   (10, self.img.shape[0] // 2), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv.imshow(nom_fenetre, ecran_attente)
        cv.waitKey(10)

    def lire_valeur_glissiere(self):
        nouvelle_valeur = cv.getTrackbarPos("Classes", self.nom_fichier)
        if nouvelle_valeur < self.nb_classes:
            self.meilleur_labels = None
        self.nb_classes = nouvelle_valeur
        self.iteration = cv.getTrackbarPos("Iteration", self.nom_fichier)
        self.essai = cv.getTrackbarPos("Essai", self.nom_fichier)
        nouvelle_valeur = cv.getTrackbarPos("Couleur", self.nom_fichier)
        if self.couleur != nouvelle_valeur:
            self.meilleur_labels = None
        self.couleur = nouvelle_valeur

    def preparation_interface(self):
        cv.namedWindow(self.nom_fichier)
        ajouter_glissiere("Classes", self.nom_fichier, 2, 255, self.nb_classes)
        ajouter_glissiere("Iteration", self.nom_fichier, 1, 255, self.iteration)
        ajouter_glissiere("Essai", self.nom_fichier, 1, 255, self.essai)
        ajouter_glissiere("Couleur", self.nom_fichier, 0, 1, self.couleur)

    def run(self):
        my_app = wx.App()
        self.nom_fichier = wx.FileSelector("Choose a file to open")
        self.img = cv.imread(self.nom_fichier, cv.IMREAD_COLOR)
        if self.img is None:
            print('Le fichier image ne peut etre lu')
            return
        self.preparation_interface()

        while True:
            cv.imshow(self.nom_fichier, self.img)
            code_touche_clavier = cv.waitKey(10)
            if code_touche_clavier == CODE_TOUCHE_ECHAP:
                break
            if code_touche_clavier == ord('e'):
                self.lire_valeur_glissiere()
                self.efface_fenetre("Labels")
                self.classification()
            if code_touche_clavier == ord('r'):
                self.meilleur_labels = None
                self.efface_fenetre("Labels")

if __name__ == '__main__':
    ClassifKMoyennes().run()
