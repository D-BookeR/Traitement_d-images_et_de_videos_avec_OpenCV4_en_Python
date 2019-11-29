import argparse
import threading
import time
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27

class GestionFlux(threading.Thread):
    def __init__(self, fluxvideo):
        threading.Thread.__init__(self)
        self.video = fluxvideo
        self.image_lue = True
        self.derniere_image = []
        self.pgm_en_cours = True
        self.verrou = threading.Lock()
    def run(self):
        if self.video is None or not self.video.isOpened():
            print("Le flux n est pas ouvert. Fin du thread")
            return
        while self.pgm_en_cours:
            ret, img = self.video.read()
            if ret:
                if calcul_median:
                    img = cv.medianBlur(img, ksize=2 * taille_noyau + 1)
                while not self.image_lue and self.pgm_en_cours:
                    time.sleep(0.001)
                self.verrou.acquire()
                self.derniere_image = img
                self.image_lue = False
                self.verrou.release()

def sauver_configuration(nom_fichier, webcams):
    fichier = cv.FileStorage(nom_fichier, cv.FILE_STORAGE_WRITE)
    if fichier.isOpened():
        for index, scam in enumerate(webcams):
            largeur, hauteur = lire_resolution(scam[1])
            fichier.write("cam" + str(index), np.array([scam[0], largeur, hauteur]))
        fichier.release()

def restaurer_configuration(nom_fichier):
    webcams = []
    fichier = cv.FileStorage(nom_fichier, cv.FILE_STORAGE_READ)
    if fichier.isOpened():
        index_webcam = 0
        while True:
            nom_noeud = "cam" + str(index_webcam)
            val_noeud = fichier.getNode(nom_noeud).mat()
            if val_noeud is not None:
                webcams.append(val_noeud)
                index_webcam = index_webcam  + 1
            else:
                break
        fichier.release()
    return webcams

def lire_resolution(webcam):
    largeur = webcam.get(cv.CAP_PROP_FRAME_WIDTH)
    hauteur = webcam.get(cv.CAP_PROP_FRAME_HEIGHT)
    return largeur, hauteur

def fixer_resolution(webcam, resolution):
    largeur, hauteur = resolution
    status = webcam.set(cv.CAP_PROP_FRAME_HEIGHT, hauteur)
    status = status and webcam.set(cv.CAP_PROP_FRAME_WIDTH, largeur)
    nouveau_x, nouveau_y = lire_resolution(webcam)
    return status and largeur == nouveau_x and hauteur == nouveau_y

def calcul_zoom(chaine_zoom):
    nouveau_zoom = 1.0
    if chaine_zoom:
        pos_deux_points = chaine_zoom.find(':')
        if pos_deux_points >= 1:
            num_zoom = float(chaine_zoom[:pos_deux_points])
            den_zoom = float(chaine_zoom[pos_deux_points+1:])
            if den_zoom > 0 and num_zoom > 0:
                nouveau_zoom = num_zoom / den_zoom
            else:
                print("Erreur ", num_zoom, "/", den_zoom, " : valeur non definie")
                exit()
        else:
            nouveau_zoom = float(str(chaine_zoom))
    return nouveau_zoom

class PositionFenetre:
    def __init__(self, r):
        self.rect_cam = r
        self.ind_cam_active = 0

def gestion_souris(event, souris_x, souris_y, _, param):
    if event == cv.EVENT_LBUTTONDOWN:
        for index, rect in enumerate(param.rect_cam):
            [x_gauche, y_haut, largeur, hauteur] = rect
            if x_gauche < souris_x < x_gauche + largeur and y_haut <= souris_y < y_haut + hauteur:
                param.ind_cam_active = index
                break
def ouvrir_webcam(liste_resolution):
    webcam_ouvertes = []
    if not liste_resolution:
        print("La ligne de commande est vide - Recherche camera par defaut")
        i = 0
        while True:
            webcam = cv.VideoCapture(i + cv.CAP_DSHOW)
            if webcam.isOpened():
                webcam_ouvertes.append((i, webcam))
            else:
                break
            i = i + 1
    else:
        for index, largeur, hauteur in  liste_resolution:
            webcam = cv.VideoCapture(int(index) + cv.CAP_DSHOW)
            if webcam.isOpened():
                if fixer_resolution(webcam, (largeur, hauteur)):
                    webcam_ouvertes.append((index, webcam))
                else:
                    print("Webcam ", index, "Taille incorrecte")
            else:
                print("Webcam ", index, " inacessible")

    if not webcam_ouvertes:
        print(
            "Aucun peripherique video ouvert verifiez votre systeme ou la ligne de commande"
            )
        exit()
    return webcam_ouvertes

def preparation_fenetrage_unique(webcam_ouvertes):
    taille_dst = np.array([0, 0])
    rect_dst_cam = []
    for webcam in webcam_ouvertes:
        largeur, hauteur = lire_resolution(webcam[1])
        largeur, hauteur = [zoom * largeur, zoom * hauteur]
        rect_dst_camera = [0, int(taille_dst[1]), int(largeur), int(hauteur)]
        rect_dst_cam.append(rect_dst_camera)
        taille_dst[1] = taille_dst[1] + hauteur
        taille_dst[0] = max(taille_dst[0], largeur)
    pos_fen = PositionFenetre(rect_dst_cam)
    frame_unique = np.zeros((taille_dst[1], taille_dst[0], 3), np.uint8)
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.setMouseCallback('Image', gestion_souris, pos_fen)
    return taille_dst, rect_dst_cam, pos_fen, frame_unique

def initialise_parser(parser):
    parser.add_argument('nom_fichier', nargs='?',
                        help='nom du fichier de configuration',
                        type=str
                        )
    parser.add_argument('-r', action='append', nargs=3, default=[],
                        help='index largeur hauteur pour webcam index',
                        type=list, dest='resolution'
                        )
    parser.add_argument('-z', action='store', default='1:2',
                        help='zoom pour toutes les images.  Exemple zoom 1/2  -z 1:2',
                        type=str, dest='zoom'
                        )
    parser.add_argument('-f', action='store', default=None,
                        help='sauvegarde de la configuration en xml oy yml  ',
                        type=str, dest='nom_fichier_yxml'
                        )
    parser.add_argument('-d', action='store_false', default=True,
                        help='Une webcam par fenetre'
                        )
    args = parser.parse_args()
    liste_resolution = []
    if args.nom_fichier is None:
        for cam in args.resolution:
            desc_cam = [
                np.double(''.join(cam[0])),
                np.double(''.join(cam[1])),
                np.double(''.join(cam[2]))
                ]
            liste_resolution.append(np.array(desc_cam))
    else:
        liste_resolution = restaurer_configuration(args.nomFichier)
    zoom = calcul_zoom(args.zoom)
    affichage_unique = args.d
    return args, zoom, liste_resolution, affichage_unique

parser = argparse.ArgumentParser(description='Ecran de surveillance')
args, zoom, liste_resolution, affichage_unique = initialise_parser(parser)
webcam_ouvertes = ouvrir_webcam(liste_resolution)
liste_thread = []
if affichage_unique:
    taille_dst, rect_dst_cam, pos_fen, frame_unique = preparation_fenetrage_unique(webcam_ouvertes)
dico_touche_median = {ord('M'): True, ord('m'): False}
dico_taille_noyau = {ord('+'): 1, ord('-'): -1}
calcul_median = False
taille_noyau = 15
for r in  webcam_ouvertes:
    g = GestionFlux(r[1])
    liste_thread.append(g)
for f in liste_thread:
    f.start()
tps_ini = time.clock()
nb_prises = 0
while True:
    liste_images = []
    for idx, th in enumerate(liste_thread):
        while th.image_lue:
            time.sleep(0.01)
        with th.verrou:
            liste_images.append(np.copy(th.derniere_image))
            th.image_lue = True
    for idx, img in enumerate(liste_images):
        if affichage_unique:
            if pos_fen.ind_cam_active == idx:
                cv.imshow('Image choisie', img)
            if zoom != 1:
                img = cv.resize(img, (0, 0), fx=zoom, fy=zoom)
                frame_unique[
                    rect_dst_cam[idx][1]:rect_dst_cam[idx][1]+rect_dst_cam[idx][3],
                    rect_dst_cam[idx][0]:rect_dst_cam[idx][0]+rect_dst_cam[idx][2]
                    ] = img
        else:
            if zoom != 1:
                img = cv.resize(img, (0, 0), fx=zoom, fy=zoom)
            cv.imshow('Image ' + str(idx), img)
        if affichage_unique:
            cv.imshow('Image', frame_unique)
    code_touche_clavier = cv.waitKey(1)
    if code_touche_clavier == CODE_TOUCHE_ECHAP:
        break
    if code_touche_clavier in dico_touche_median:
        calcul_median = dico_touche_median[code_touche_clavier]
        nb_prises = 0
        tps_ini = time.clock()
    if code_touche_clavier in dico_taille_noyau:
        taille_noyau = taille_noyau+dico_taille_noyau[code_touche_clavier]
        if taille_noyau < 0:
            taille_noyau = 0
        nb_prises = 0
        print("taille noyau = ", taille_noyau)
        tps_ini = time.clock()
    nb_prises = nb_prises + len(webcam_ouvertes)
    if nb_prises > 100:
        nb_images_par_seconde = nb_prises / (time.clock() - tps_ini)
        print("Pour chaque camera : ", nb_images_par_seconde, " Images par seconde")
        if calcul_median:
            print("     pour un Noyau = ", taille_noyau)
        nb_prises = 0
        tps_ini = time.clock()
for p in liste_thread:
    with p.verrou:
        p.pgm_en_cours = False
time.sleep(0.5)

if args.nom_fichier_yxml is not None:
    sauver_configuration(args.nom_fichier_yxml, webcam_ouvertes)
for cam in webcam_ouvertes:
    cam[1].release()
cv.destroyAllWindows()
