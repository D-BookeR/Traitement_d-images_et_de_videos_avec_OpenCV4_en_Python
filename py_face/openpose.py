import random
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
CAMAPI = cv.CAP_DSHOW
PATH_MODELE = "f:/testDNN/openpose/pose/body_25/"

try:
    import wx
    mon_appli = wx.App()
except ImportError:
    mon_appli = None
ARTICULATION = ["Nez", "Cou",
                "EpauleD", "CoudeD", "PoignetD",
                "EpauleG", "CoudeG", "PoignetG",
                "Bassin",
                "HancheD", "GenouD", "ChevilleD",
                "HancheG", "GenouG", "ChevilleG",
                "OeilD", "OeilG",
                "OreilleD", "OreilleG",
                "GrosOrteilG", "PetitOrteilG", "talonG",
                "GrosOrteilD", "PetitOrteilD", "talonD",
                "Fond"
                ]
LIENS = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4],
    [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
    [8, 12], [12, 13], [13, 14], [1, 0], [0, 15],
    [15, 17], [0, 16], [16, 18], [2, 17], [5, 18],
    [14, 19], [19, 20], [14, 21], [11, 22], [22, 23],
    [11, 24]
    ]
CARTE = [
    [0, 1], [14, 15], [22, 23], [16, 17], [18, 19],
    [24, 25], [26, 27], [6, 7], [2, 3], [4, 5],
    [8, 9], [10, 11], [12, 13], [30, 31], [32, 33],
    [36, 37], [34, 35], [38, 39], [20, 21], [28, 29],
    [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
    [50, 51]
    ]

def sauver_inference(out):
    i = 0
    for img in out[0, :, :, :]:
        cv.imwrite('f:/tmp/out' + str(i) + '.png', np.absolute(img*100))
        i = i + 1
def tracer_articulations(img, idx_art, liste_points, zoom):
    (zoom_x, zoom_y) = zoom
    for idx_part,idx in enumerate(idx_art):
        for i in idx:
            pts_art = liste_points[i]
            x = int(pts_art[0] * zoom_x)
            y = int(pts_art[1] * zoom_y)
            cv.circle(img, (x, y), 3, (0, 0, 255))
            cv.putText(img, str(idx_part),(x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    return img

def selection_points(sortie):
    ind_point = 0
    liste_points = []
    idx_art = []
    for idx in range(len(ARTICULATION)-1):
        idx_pts = []
        art = sortie[0, idx, :, :] > 0.08
        nb_art, _, stats, _ = cv.connectedComponentsWithStats(art.astype(np.uint8))
        for i in range(0, nb_art):
            if stats[i, 4] < 100:
                pos_masse = np.mgrid[-stats[i, 3] / 2 + 0.5 :stats[i, 3] / 2,
                                     -stats[i, 2] / 2 + 0.5 :stats[i, 2] / 2]
                poids = sortie[0, idx,
                            stats[i, 1]:stats[i, 1]+stats[i, 3],
                            stats[i, 0]:stats[i, 0]+stats[i, 2]]
                poids_total = np.sum(poids)
                gravite_x = np.sum(poids * pos_masse[1, :, :]) / poids_total
                gravite_y = np.sum(poids * pos_masse[0, :, :]) / poids_total
                pos_art = (gravite_x + stats[i, 0] + stats[i, 2] / 2,
                           gravite_y + stats[i, 1] + stats[i, 3] / 2)
                idx_pts.append(ind_point)
                liste_points.append(pos_art)
                ind_point = ind_point + 1
        idx_art.append(idx_pts)
    return liste_points, idx_art

def detection_personne(out, idx_art):
    liste_personne = []
    for lien, carte in zip(LIENS, CARTE):
        for idx1 in idx_art[lien[0]]:
            pts1 = liste_points[idx1]
            paire = []
            dmin_paire = 1000
            for idx2 in idx_art[lien[1]]:
                pts2 = liste_points[idx2]
                u = np.array([pts2[0] - pts1[0], pts2[1] - pts1[1]])
                u2 = np.linalg.norm(u)
                u = u / u2
                pts_lien = list(zip(np.linspace(pts1[0], pts2[0], num=10),
                                    np.linspace(pts1[1], pts2[1], num=10)))
                vecteur_colin = 0
                for pts in pts_lien:
                    x, y = int((pts[0])), int((pts[1]))
                    v = np.array([out[0, len(ARTICULATION) + carte[0], y, x],
                                  out[0, len(ARTICULATION) + carte[1], y, x]])
                    if np.dot(u, v) > 0.2:
                        vecteur_colin = vecteur_colin + 1
                if vecteur_colin >= 6:
                    if u2 < dmin_paire:
                        dmin_paire = u2
                        paire = [idx1, idx2]

            if len(paire) != 0:
                pt_insere = True
                for graphe_personne in liste_personne:
                    if paire[0] in graphe_personne or paire[1] in graphe_personne:
                        graphe_personne.append(paire[0])
                        graphe_personne.append(paire[1])
                        pt_insere = False
                if pt_insere:
                    graphe_personne = [paire[0], paire[1]]
                    liste_personne.append(graphe_personne)
    return liste_personne

if __name__ == '__main__':
    net = cv.dnn.readNetFromCaffe(PATH_MODELE + "pose_deploy.prototxt",
                                  PATH_MODELE + "pose_iter_584000.caffemodel")
    nom_fenetre = "Image"
    cv.namedWindow("Image")
    code = ord('o')
    while code != CODE_TOUCHE_FIN:
        if code == ord('o'):
            if mon_appli is None:
                nom_image = 'f:/testdnn/objectdetection/Images/ScavengherHunt.jpg'
                img_ref = cv.imread(nom_image)
            else:
                nom_image = wx.FileSelector(
                    "Image",
                    wildcard="image jpeg  (*.jpg)|*.jpg|image tiff  (*.tif)|*.tif")
                img_ref = cv.imread(nom_image)
        if img_ref is not None:
            hauteur_blob = 368
            largeur_blob = int((hauteur_blob * img_ref.shape[1]) / img_ref.shape[0])
            reste = largeur_blob % 16
            if reste != 0:
                if reste <= 8:
                    largeur_blob = largeur_blob - reste
                else:
                    largeur_blob = largeur_blob + 16 - reste

            inp = cv.dnn.blobFromImage(img_ref, 1/255, (largeur_blob, hauteur_blob),
                                       (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inp)
            sortie = net.forward()
            img = img_ref.copy()
            sauver_inference(sortie)
            _, nb_part, nb_lig, nb_col = sortie.shape
            zoom_x = img_ref.shape[1] / nb_col
            zoom_y = img_ref.shape[0] / nb_lig
            liste_points, idx_art = selection_points(sortie)
            liste_personne = detection_personne(sortie, idx_art)
            img = tracer_articulations(img, idx_art, liste_points, (zoom_x, zoom_y))
            for personne in liste_personne:
                couleur = (random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255))
                for idx in range(0, len(personne), 2):
                    p1 = liste_points[personne[idx]]
                    p2 = liste_points[personne[idx + 1]]
                    cv.line(img, (int(p1[0] * zoom_x), int(p1[1] * zoom_y)),
                            (int(p2[0] * zoom_x), int(p2[1] * zoom_y)), couleur, 2)
                cv.imshow("Image", img_ref)
                cv.imshow("Graphe", img)
            cv.imshow("Image", img_ref)
            cv.imshow("Graphe", img)
        code = cv.waitKey()

    cv.destroyAllWindows()
    exit()
