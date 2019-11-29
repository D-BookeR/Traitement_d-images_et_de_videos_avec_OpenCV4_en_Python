import time
import threading
import datetime
import numpy as np
import vtk
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
MODE_AFFICHAGE = np.int(0x100)
MODE_REGLAGECAMERA = np.int(0x1000)
NBCAMERA = 2
CAMAPI = cv.CAP_DSHOW

LARGEUR_ECRAN = 1200
HAUTEUR_ECRAN = 900

MODE_CALIBRE = np.int(0x200)
MODE_MAstereo = np.int(0x400)
MODE_EPIPOLAIRE = np.int(0x800)

MODE_3D = np.int(0x2000)

class ParamAffichage:
    def __init__(self):
        self.index_camera = 0
        self.mode_affichage = MODE_AFFICHAGE
        self.frame = []
        self.zoom = 1
        self.code_touche_clavier = 0
        self.algo_stereo = 0
        self.type_distorsion = 0
        self.taille_globale = [0, 0]
    def init_fenetre(self, liste_webcam_ouvertes, liste_cam_calib):
        taille_webcam = []
        for idx, vid in enumerate(liste_webcam_ouvertes):
            ret = fixer_resolution(vid[0], (liste_cam_calib[idx].taille_img[0],
                                            liste_cam_calib[idx].taille_img[1]))
            if not ret:
                print("La resolution de la camera ", idx, "ne peut être fixée\nFin du programme")
                exit()

            taille_webcam.append((int(liste_cam_calib[idx].taille_img[0]),
                                  int(liste_cam_calib[idx].taille_img[1])))
            self.taille_globale[0] += taille_webcam[idx][0]
            self.taille_globale[1] = int(max(liste_cam_calib[idx].taille_img[1],
                                             self.taille_globale[1]))
        if self.taille_globale[0] > LARGEUR_ECRAN or self.taille_globale[1] > HAUTEUR_ECRAN:
            self.zoom = min(LARGEUR_ECRAN/self.taille_globale[0],
                            HAUTEUR_ECRAN/self.taille_globale[1])
        self.frame = np.zeros((self.taille_globale[1], self.taille_globale[0], 3),
                              np.uint8)
        return taille_webcam

class CalibrageCamera:
    def __init__(self):
        self.idx_usb = 0
        self.index = 0
        self.rvecs = None
        self.tvecs = None
        self.fic_donnees = None
        self.mat_intrin = None
        self.mat_dist = None
        self.pts_grille = []
        self.pts_camera = []
        self.pts_objets = []
        self.taille_img = (640, 480)
        self.rms = 0
        self.type_calib = [
            cv.CALIB_FIX_PRINCIPAL_POINT+cv.CALIB_FIX_K1+\
            cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4+
            cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6,
            cv.CALIB_USE_INTRINSIC_GUESS
            ]

    def calibrer(self, taille):
        if self.fic_donnees is None:
            sauver_donnees_calibration(self, None, None)
        if self.mat_intrin is None:
            self.type_calib[0] = 0
        else:
            self.type_calib[0] = cv.CALIB_USE_INTRINSIC_GUESS
        for param_otim in self.type_calib:
            self.rms, self.mat_intrin, self.mat_dist, \
            self.rvecs, self.tvecs = \
            cv.calibrateCamera(self.pts_objets,
                               self.pts_camera,
                               taille,
                               self.mat_intrin,
                               self.mat_dist,
                               self.rvecs,
                               self.tvecs,
                               param_otim,
                               (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 5000000, 1e-8))
        self.type_calib[0] = self.type_calib[0] | cv.CALIB_USE_EXTRINSIC_GUESS

    def recherche_grille(self, img, mire):
        grille, echiquier = cv.findChessboardCorners(
            img, (mire.nb_col, mire.nb_lig),
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK
            )
        if grille:
            img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            echiquier = cv.cornerSubPix(img_gris, echiquier, (5, 5), (-1, -1),
                                        (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 100, 0.01)
                                        )
        else:
            return False, []
        return True, echiquier

    def analyse_grille(self, img, frame, mire):
        grille, echiquier = self.recherche_grille(img, mire)
        if grille:
            cv.drawChessboardCorners(frame, (mire.nb_col, mire.nb_lig), echiquier, False)
            self.pts_objets.append(self.pts_grille)
            self.pts_camera.append(echiquier)
        else:
            return False, frame
        return True, frame

    def analyse_charuco(self, img, frame, mire):
        pts_aruco, ids, refus = cv.aruco.detectMarkers(image=img, dictionary=mire.dictionary,
                                                       parameters=mire.detectorParams)
        pts_aruco, ids, refus, recover = cv.aruco.refineDetectedMarkers(
            img, mire.board,
            pts_aruco, ids,
            rejectedCorners=refus)
        if recover is not None:
            pts_aruco = permutation_taille(pts_aruco)

        if ids is not None:
            retval, coin_charuco, ids_charuco = cv.aruco.interpolateCornersCharuco(
                pts_aruco, ids, img, mire.board)
            if retval == 0:
                return False, frame, mire
            cv.aruco.drawDetectedCornersCharuco(
                frame, coin_charuco, ids_charuco)
            cv.aruco.drawDetectedMarkers(frame, pts_aruco, ids)
            pts_reel = None
            echiquier = None
            if False:
                for ir in range(len(pts_aruco)):
                    if echiquier is None:
                        echiquier = pts_aruco[ir][0]
                        pts_reel = mire.board.objPoints[ids[ir, 0]]
                    else:
                        echiquier = np.vstack((echiquier, pts_aruco[ir][0]))
                        pts_reel = np.vstack((pts_reel, mire.board.objPoints[ids[ir, 0]]))
            if True:
                for ir in range(ids_charuco.shape[0]):
                    index = ids_charuco[ir][0]
                    if echiquier is None:
                        echiquier = coin_charuco[ir][0]
                        pts_reel = mire.board.chessboardCorners[index]
                    else:
                        echiquier = np.vstack((echiquier, coin_charuco[ir][0]))
                        pts_reel = np.vstack((pts_reel, mire.board.chessboardCorners[index]))
            pts_reel = pts_reel.reshape((pts_reel.shape[0], 1, 3))
            self.pts_objets.append(pts_reel)
            self.pts_camera.append(echiquier)
        else:
            return False, frame
        return True, frame


class ParamMire:
    def __init__(self):
        self.nb_lig = 6
        self.nb_col = 9
        self.nb_lig_aruco = 5
        self.nb_col_aruco = 8
        self.dim_carre = 0.0275
        self.dim_aruco = 0.03455
        self.sep_aruco = 0.02164
        self.dict = cv.aruco.DICT_7X7_250
        self.dictionary = cv.aruco.Dictionary_get(self.dict)
        self.board =	cv.aruco.CharucoBoard_create(self.nb_col_aruco, self.nb_lig_aruco,
                                                  self.dim_aruco, self.sep_aruco,
                                                  self.dictionary)
        self.detectorParams = cv.aruco.DetectorParameters_create()
        self.detectorParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

class ParamAlgoStereo:
    def __init__(self):
        self.al_bm = None
        self.al_sgbm = None
        self.pre_filtre = cv.StereoBM_PREFILTER_XSOBEL
        self.taille_bloc = 3
        self.nb_disparite = 8
        self.etendu_speck = 1
        self.taille_speck = 10
        self.unicite = 3
        self.pre_filtre_cap = 31
    def maj_stereo_bloc(self, x):
        self.taille_bloc = x
        self.al_bm.setBlockSize(2*self.taille_bloc+1)
        self.al_sgbm.setBlockSize(2 * self.taille_bloc + 1)
    def maj_stereo_disp(self, x):
        self.nb_disparite = x
        self.al_bm.setNumDisparities(16*self.nb_disparite)
        self.al_sgbm.setNumDisparities(16*self.nb_disparite)
    def maj_stereo_unicite(self, x):
        self.unicite = x
        self.al_bm.setUniquenessRatio(self.unicite)
        self.al_sgbm.setUniquenessRatio(self.unicite)
    def maj_stereo_taille_speck(self, x):
        self.taille_speck = x
        self.al_bm.setSpeckleWindowSize(self.taille_speck)
        self.al_sgbm.setSpeckleWindowSize(self.taille_speck)
    def maj_stereo_etendu_speck(self, x):
        self.etendu_speck = x
        self.al_bm.setSpeckleRange(self.etendu_speck)
        self.al_sgbm.setSpeckleRange(self.etendu_speck)

class CalibrageCamera3D:
    def __init__(self):
        self.type_calib = [cv.CALIB_FIX_INTRINSIC + cv.CALIB_ZERO_DISPARITY,
                           cv.CALIB_ZERO_DISPARITY+cv.CALIB_USE_INTRINSIC_GUESS]
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.m = None
        self.d = None
        self.epipolar = None
        self.valid1 = ()
        self.valid2 = ()
        self.rms = 0
        self.fic_donnees = None
        self.pts_grille = []
        self.pts_camera_gauche = []
        self.pts_camera_droite = []
        self.pts_objets = []
        self.map11 = None
        self.map12 = None
        self.map21 = None
        self.map22 = None
    def analyse_grille(self, liste_img, frame, mire):
        grille, echiquier_gauche = CalibrageCamera().recherche_grille(liste_img[0],mire)
        if grille: 
            grille, echiquier_droite = CalibrageCamera().recherche_grille(liste_img[1],mire)
            if grille:
                self.pts_objets.append(np.asarray(self.pts_grille, dtype=np.float32))
                self.pts_camera_gauche.append(echiquier_gauche)
                self.pts_camera_droite.append(echiquier_droite)
            else:
                return False, frame
        else:
            return False, frame
        largeur, hauteur = liste_img[0].shape[1], liste_img[0].shape[0]
        cv.drawChessboardCorners(frame[0:hauteur, 0:largeur], 
                                 (mire.nb_col, mire.nb_lig), echiquier_gauche, False)
        cv.drawChessboardCorners(frame[0:hauteur, largeur:2 * largeur], 
                                 (mire.nb_col, mire.nb_lig), echiquier_droite, False)
        return True, frame
        echiquier = []
        grille, echiquier = cv.findChessboardCorners(
            img, (mire.nb_col, mire.nb_lig),
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
            )
        if grille:
            img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            echiquier = cv.cornerSubPix(
                img_gris, echiquier, (5, 5), (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 100, 0.01))
            cv.drawChessboardCorners(frame, (mire.nb_col, mire.nb_lig), echiquier, False)
            if index == 0:
                if self.pts_camera_gauche is None:
                    self.pts_objets = [np.asarray(self.pts_grille, dtype=np.float32)]
                    self.pts_camera_gauche = [echiquier]
                else:
                    self.pts_objets.append(np.asarray(self.pts_grille, dtype=np.float32))
                    self.pts_camera_gauche.append(echiquier)
            else:
                if self.pts_camera_droite is None:
                    self.pts_camera_droite = [echiquier]
                else:
                    self.pts_camera_droite.append(echiquier)
        else:
            return False, frame
        return True, frame

    def  analyse_charuco(self, liste_images, mire, param_ecran):
        pts_arucog, idsg, refusg = cv.aruco.detectMarkers(
            image=liste_images[0], dictionary=mire.dictionary,
            parameters=mire.detectorParams)
        pts_arucog, idsg, refusg, recoverg = cv.aruco.refineDetectedMarkers(
            liste_images[0], mire.board,
            pts_arucog, idsg, rejectedCorners=refusg)
        if recoverg is not None:
            pts_arucog = permutation_taille(pts_arucog)
        pts_arucod, idsd, refusd = cv.aruco.detectMarkers(
            liste_images[1], dictionary=mire.dictionary,
            parameters=mire.detectorParams)
        pts_arucod, idsd, refusd, recoverd = cv.aruco.refineDetectedMarkers(
            liste_images[1], mire.board,
            pts_arucod, idsd, rejectedCorners=refusd)
        if recoverd is not None:
            pts_arucod = permutation_taille(pts_arucod)
        if len(idsg) > 0 and len(idsd) > 0:
            frame = param_ecran.frame
            cv.aruco.drawDetectedMarkers(param_ecran.frame, pts_arucog, idsg)
            r_dst = [liste_images[1].shape[1], 0,
                     liste_images[1].shape[1], liste_images[1].shape[0]]
            y = np.zeros((liste_images[1].shape[0], liste_images[1].shape[1], 3),
                         np.uint8)
            cv.aruco.drawDetectedMarkers(y, pts_arucod, idsd)
            frame[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = y + frame[
                r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]
            param_ecran.frame = frame
            pts_reel = None
            for ir in range(len(pts_arucog)):
                ind = np.where(idsd == idsg[ir])
                if ind[0].shape[0] > 0:
                    id = ind[0][0]
                    if pts_reel is None:
                        echiquierg = pts_arucog[ir][0]
                        echiquierd = pts_arucod[id][0]
                        pts_reel = mire.board.objPoints[idsg[ir, 0]]
                    else:
                        echiquierg = np.vstack((echiquierg, pts_arucog[ir][0]))
                        echiquierd = np.vstack((echiquierd, pts_arucod[id][0]))
                        pts_reel = np.vstack((pts_reel, mire.board.objPoints[idsg[ir, 0]]))
            pts_reel = pts_reel.reshape((pts_reel.shape[0], 1, 3))
            self.pts_objets.append(pts_reel)
            self.pts_camera_gauche.append(echiquierg)
            self.pts_camera_droite.append(echiquierd)

    def calibrer(self, liste_cam_calib):
        sauver_donnees_calibration(None, None, self)
        self.m = [[], []]
        self.d = [[], []]
        self.m[0] = liste_cam_calib[0].mat_intrin.copy()
        self.m[1] = liste_cam_calib[1].mat_intrin.copy()
        self.d[0] = liste_cam_calib[0].mat_dist.copy()
        self.d[1] = liste_cam_calib[1].mat_dist.copy()
        taille = liste_cam_calib[0].taille_img
        for param in self.type_calib:
            resultat = cv.stereoCalibrate(
                self.pts_objets,
                self.pts_camera_gauche,
                self.pts_camera_droite,
                self.m[0], self.d[0], self.m[1], self.d[1],
                taille, self.R, self.T, self.E, self.F,
                param,
                (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 5000000, 1e-8))
            self.rms, self.m[0], self.d[0], self.m[1],\
               self.d[1], self.R, self.T, self.E, self.F = resultat
        if self.R is not None:
            resultat = cv.stereoRectify(self.m[0], self.d[0],
                                        sys3d.m[1], sys3d.d[1],
                                        taille, self.R,
                                        self.T, self.R1,
                                        self.R2, self.P1,
                                        self.P2, self.Q,
                                        cv.CALIB_ZERO_DISPARITY, -1)
            self.R1, self.R2, self.P1, self.P2, self.Q, self.valid1, self.valid2 = resultat
        return

    def erreur_droite_epipolaire(self):
        err = 0
        nb_points = 0
        for idx, pt in enumerate(sys3d.pts_camera_gauche):
            nb_pt = len(sys3d.pts_camera_gauche[idx])
            pt1 = cv.undistortPoints(pt, sys3d.m[0], sys3d.d[0], None, sys3d.m[0])
            pt2 = cv.undistortPoints(sys3d.pts_camera_droite[idx], sys3d.m[1],
                                     sys3d.d[1], None, sys3d.m[1])
            droite1 = cv.computeCorrespondEpilines(pt1, 1, sys3d.F)
            droite2 = cv.computeCorrespondEpilines(pt2, 2, sys3d.F)
            for j in range(nb_pt):
                errij1 = np.abs(pt1[j, 0, 0] * droite2[j, 0, 0] +
                                pt1[j, 0, 1] * droite2[j, 0, 1] + droite2[j, 0, 2])
                errij2 = np.abs(pt2[j, 0, 0] * droite1[j, 0, 0] +
                                pt2[j, 0, 1]*droite1[j, 0, 1] + droite1[j, 0, 2])
                err += errij1 + errij2
            nb_points += nb_pt
        return err / nb_points / 2

def permutation_taille(pts_aruco):
    for ir in range(len(pts_aruco)):
        d1, d2, d3 = pts_aruco[ir].shape
        pts_aruco[ir] = pts_aruco[ir].reshape((d2, d1, d3))
    return pts_aruco


def lire_images(liste_thread):
    liste_images = []
    nb_image_lue = 0
    while nb_image_lue != len(liste_thread):
        nb_image_lue = 0
        for thread in liste_thread:
            with thread.verrou:
                if not thread.image_lue:
                    nb_image_lue = nb_image_lue + 1

    for thread in liste_thread:
        with thread.verrou:
            image = thread.get_image()
            if image is None:
                print("Image vide")
                return []
            liste_images.append(image)
            thread.image_lue = True
    return liste_images

def zoom(x, w):
    if w != 1:
        y = cv.resize(src=x, dsize=None, fx=w, fy=w)
        return y
    return x

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

def lire_noeud(noeud):
    if noeud.isSeq():
        l = []
        if noeud.at(0).isString():
            for i in range(0, noeud.size()):
                l.append(noeud.at(i).string())
        if noeud.at(0).isReal():
            for i in range(0, noeud.size()):
                l.append(noeud.at(i).real())
        if noeud.at(0).isInt():
            for i in range(0, noeud.size()):
                l.append(int(noeud.at(i).real()))
    else:
        if noeud.isString():
            l = noeud.string()
        elif noeud.isReal():
            l = noeud.real()
        elif noeud.isInt():
            l = int(noeud.real())
        else:
            l = noeud.mat()
    return l

def recherche_camera(api=CAMAPI):
    liste_webcam_ouvertes = []
    for i in range(NBCAMERA):
        webcam = cv.VideoCapture(i + api)
        if webcam.isOpened():
            liste_webcam_ouvertes.append((webcam, i))
    return liste_webcam_ouvertes

def install_threads(liste_webcam_ouvertes):
    liste_thread = []
    for webcam in liste_webcam_ouvertes:
        liste_thread.append(GestionFlux(webcam[0], webcam[1]))
    for thread in liste_thread:
        thread.start()
    return liste_thread

def charger_donnees_calibration(nom_fichier, pc, sys3d):
    fichier = cv.FileStorage(nom_fichier, cv.FILE_STORAGE_READ)
    if not fichier.isOpened():
        return False, pc, sys3d
    if pc is not None:
        if fichier.getNode("nbGrille") is not None:
            nb_img_grille = lire_noeud(fichier.getNode("nbGrille"))
        else:
            nb_img_grille = 0
        for idx in range(nb_img_grille):
            noeud = fichier.getNode("Grille"+str(idx))
            if noeud is not None:
                pc.pts_camera.append(lire_noeud(noeud))
            noeud = fichier.getNode("Objet"+str(idx))
            if noeud is not None:
                pc.pts_objets.append(lire_noeud(noeud))
        fichier.release()
        return True, pc, None
    elif sys3d is not None:
        if fichier.getNode("nbGrille") is not None:
            nb_grille = lire_noeud(fichier.getNode("nbGrille"))
        else:
            nb_grille = 0
        for idx in range(nb_grille):
            noeud = fichier.getNode("GrilleG"+str(idx))
            if  noeud is not None:
                sys3d.pts_camera_gauche.append(lire_noeud(noeud))
            noeud = fichier.getNode("GrilleD" + str(idx))
            if  noeud is not None:
                sys3d.pts_camera_droite.append(lire_noeud(noeud))
            noeud = fichier.getNode("Objets" + str(idx))
            if  noeud is not None:
                sys3d.pts_objets.append(lire_noeud(noeud))
        fichier.release()
        return True, None, sys3d
    fichier.release()
    return False, None, None

def charger_configuration(nom_fichier):
    sys3d = CalibrageCamera3D()
    suivi_dst = SuiviDistance()
    stereo = ParamAlgoStereo()
    cam_calib = []
    mire = ParamMire()
    cam_calib.append(CalibrageCamera())
    cam_calib.append(CalibrageCamera())
    cam_calib[0].index = 0
    cam_calib[1].index = 1
    cam_calib[0].idx_usb = 0
    cam_calib[1].idx_usb = 1
    fichier = cv.FileStorage(nom_fichier, cv.FILE_STORAGE_READ)
    if not fichier.isOpened():
        fichier.release()
        return False, mire, cam_calib, sys3d, stereo
    if not fichier.getNode("Echiquiernb_lig").empty():
        mire.nb_lig = lire_noeud(fichier.getNode("Echiquiernb_lig"))
    if not fichier.getNode("Echiquiernb_col").empty():
        mire.nb_col = lire_noeud(fichier.getNode("Echiquiernb_col"))
    if not fichier.getNode("Echiquierdim_carre").empty():
        mire.dim_carre = lire_noeud(fichier.getNode("Echiquierdim_carre"))
    if not fichier.getNode("Aruconb_lig").empty():
        mire.nb_lig_aruco = lire_noeud(fichier.getNode("Aruconb_lig"))
    if not fichier.getNode("Aruconb_col").empty():
        mire.nb_col_aruco = lire_noeud(fichier.getNode("Aruconb_col"))
    if not fichier.getNode("ArucoDim").empty():
        mire.dim_aruco = lire_noeud(fichier.getNode("ArucoDim"))
    if not fichier.getNode("ArucoSep").empty():
        mire.sep_aruco = lire_noeud(fichier.getNode("ArucoSep"))
    if not fichier.getNode("ArucoDict").empty():
        mire.dict = lire_noeud(fichier.getNode("ArucoDict"))
    if not fichier.getNode("Cam0index").empty():
        cam_calib[0].idx_usb = lire_noeud(fichier.getNode("Cam0index"))
    if not fichier.getNode("Cam0Size").empty():
        cam_calib[0].taille_img = lire_noeud(fichier.getNode("Cam0Size"))
    if not fichier.getNode("Cam1Size").empty():
        cam_calib[1].taille_img = lire_noeud(fichier.getNode("Cam1Size"))
    if not fichier.getNode("Cam1index").empty():
        cam_calib[1].idx_usb = lire_noeud(fichier.getNode("Cam1index"))
    if not fichier.getNode("type_calib1").empty():
        cam_calib[1].type_calib = lire_noeud(fichier.getNode("type_calib1"))
        if type(cam_calib[1].type_calib) is np.ndarray:
            nb_element = cam_calib[1].type_calib.size
            cam_calib[1].type_calib = list(cam_calib[1].type_calib.reshape(nb_element))
        else:
            param_optim = cam_calib[1].type_calib
            cam_calib[1].type_calib = []
            cam_calib[1].type_calib.append(param_optim)
    if not fichier.getNode("type_calib0").empty():
        cam_calib[0].type_calib = lire_noeud(fichier.getNode("type_calib0"))
        if type(cam_calib[0].type_calib) is np.ndarray:
            nb_element = cam_calib[0].type_calib.size
            cam_calib[0].type_calib = list(cam_calib[0].type_calib.reshape(nb_element))
        else:
            param_optim = cam_calib[0].type_calib
            cam_calib[0].type_calib = []
            cam_calib[0].type_calib.append(param_optim)
    if not fichier.getNode("type_calib").empty():
        sys3d.type_calib = lire_noeud(fichier.getNode("type_calib"))
        if type(sys3d.type_calib) is np.ndarray:
            nb_element = sys3d.type_calib.size
            sys3d.type_calib = list(sys3d.type_calib.reshape(nb_element))
        else:
            param_optim = sys3d.type_calib
            sys3d.type_calib = []
            sys3d.type_calib.append(param_optim)
    if not fichier.getNode("cameraMatrice0").empty():
        cam_calib[0].mat_intrin = fichier.getNode("cameraMatrice0").mat()
    if not fichier.getNode("cameraDistorsion0").empty():
        cam_calib[0].mat_dist = fichier.getNode("cameraDistorsion0").mat()
    if not fichier.getNode("cameraMatrice1").empty():
        cam_calib[1].mat_intrin = fichier.getNode("cameraMatrice1").mat()
    if not fichier.getNode("cameraDistorsion1").empty():
        cam_calib[1].mat_dist = fichier.getNode("cameraDistorsion1").mat()
    if not fichier.getNode("ficDonnee0").empty():
        cam_calib[0].fic_donnees = lire_noeud(fichier.getNode("ficDonnee0"))
        if cam_calib[0].fic_donnees is not None:
            f, p, s = charger_donnees_calibration(cam_calib[0].fic_donnees, cam_calib[0], None)
            if f:
                cam_calib[0] = p
    if not fichier.getNode("ficDonnee1").empty():
        cam_calib[1].fic_donnees = lire_noeud(fichier.getNode("ficDonnee1"))
        if cam_calib[1].fic_donnees is not None:
            f, p, s = charger_donnees_calibration(cam_calib[1].fic_donnees, cam_calib[1], None)
            if f:
                cam_calib[1] = p
    if not fichier.getNode("ficDonnee3d").empty():
        sys3d.fic_donnees = lire_noeud(fichier.getNode("ficDonnee3d"))
        if sys3d.fic_donnees is not None:
            f, p, s = charger_donnees_calibration(sys3d.fic_donnees, None, sys3d)
            if f:
                sys3d = s
    if not fichier.getNode("R").empty():
        sys3d.R = fichier.getNode("R").mat()
        sys3d.T = fichier.getNode("T").mat()
        sys3d.R1 = fichier.getNode("R1").mat()
        sys3d.R2 = fichier.getNode("R2").mat()
        sys3d.P1 = fichier.getNode("P1").mat()
        sys3d.P2 = fichier.getNode("P2").mat()
        sys3d.Q = fichier.getNode("Q").mat()
        sys3d.F = fichier.getNode("F").mat()
        sys3d.E = fichier.getNode("E").mat()
        sys3d.valid1 = lire_noeud(fichier.getNode("rect1"))
        sys3d.valid2 = lire_noeud(fichier.getNode("rect2"))
        sys3d.m = [np.array((1, 1), np.uint8), np.array((1, 1), np.uint8)]
        sys3d.d = [np.array((1, 1), np.uint8), np.array((1, 1), np.uint8)]
        sys3d.m[0] = fichier.getNode("M0").mat()
        sys3d.d[0] = fichier.getNode("D0").mat()
        sys3d.m[1] = fichier.getNode("M1").mat()
        sys3d.d[1] = fichier.getNode("D1").mat()
    if not fichier.getNode("algo_sgbm").empty():
        stereo.pre_filtre = lire_noeud(fichier.getNode("preFilterType"))
        stereo.pre_filtre_cap = lire_noeud(fichier.getNode("pre_filtre_cap"))
        stereo.taille_bloc = lire_noeud(fichier.getNode("blockSize"))
        stereo.nb_disparite = lire_noeud(fichier.getNode("numDisparities"))
        stereo.unicite = lire_noeud(fichier.getNode("uniquenessRatio"))
        stereo.etendu_speck = lire_noeud(fichier.getNode("speckleRange"))
        stereo.taille_speck = lire_noeud(fichier.getNode("speckleSize"))
    return True, mire, cam_calib, sys3d, stereo

def sauver_configuration(nom_fichier, mire, cam_calib, sys3d, stereo):
    d = datetime.datetime.now()
    nom_fichier = nom_fichier+str(d)+".yml"
    nom_fichier = nom_fichier.replace(':', '_')
    nom_fichier = nom_fichier.replace('-', '_')
    fichier = cv.FileStorage(nom_fichier, cv.FILE_STORAGE_WRITE)
    if not fichier.isOpened():
        return
    fichier.write("date", str(d))
    fichier.write("Echiquiernb_lig", mire.nb_lig)
    fichier.write("Echiquiernb_col", mire.nb_col)
    fichier.write("Echiquierdim_carre", mire.dim_carre)
    fichier.write("Aruconb_lig", mire.nb_lig_aruco)
    fichier.write("Aruconb_col", mire.nb_col_aruco)
    fichier.write("ArucoDim", mire.dim_aruco)
    fichier.write("ArucoSep", mire.sep_aruco)
    fichier.write("ArucoDict", mire.dict)
    fichier.write("type_calib0", np.array(cam_calib[0].type_calib))
    fichier.write("Cam0index", cam_calib[0].idx_usb)
    fichier.write("Cam0Size", np.array(cam_calib[0].taille_img))
    fichier.write("type_calib1", np.array(cam_calib[1].type_calib))
    fichier.write("Cam1index", cam_calib[1].idx_usb)
    fichier.write("Cam1Size", np.array(cam_calib[1].taille_img))
    if cam_calib[0].mat_intrin is not None:
        fichier.write("cameraMatrice0", cam_calib[0].mat_intrin)
        fichier.write("cameraDistorsion0", cam_calib[0].mat_dist)
        fichier.write("ficDonnee0", cam_calib[0].fic_donnees)
    if cam_calib[1].mat_intrin is not None:
        fichier.write("cameraMatrice1", cam_calib[1].mat_intrin)
        fichier.write("cameraDistorsion1", cam_calib[1].mat_dist)
        fichier.write("ficDonnee1", cam_calib[1].fic_donnees)
    if sys3d.R is not None:
        fichier.write("ficDonnee3d", sys3d.fic_donnees)
        fichier.write("type_calib", np.array(sys3d.type_calib))
        fichier.write("R", sys3d.R)
        fichier.write("T", sys3d.T)
        fichier.write("R1", sys3d.R1)
        fichier.write("R2", sys3d.R2)
        fichier.write("P1", sys3d.P1)
        fichier.write("P2", sys3d.P2)
        fichier.write("Q", sys3d.Q)
        fichier.write("F", sys3d.F)
        fichier.write("E", sys3d.E)
        fichier.write("rect1", np.array(sys3d.valid1))
        fichier.write("rect2", np.array(sys3d.valid2))
        fichier.write("M0", sys3d.m[0])
        fichier.write("D0", sys3d.d[0])
        fichier.write("M1", sys3d.m[1])
        fichier.write("D1", sys3d.d[1])
    if stereo is not None:
        fichier.write("algo_sgbm", 1)
        fichier.write("preFilterType", stereo.pre_filtre)
        fichier.write("pre_filtre_cap", stereo.pre_filtre_cap)
        fichier.write("blockSize", stereo.taille_bloc)
        fichier.write("numDisparities", stereo.nb_disparite)
        fichier.write("uniquenessRatio", stereo.unicite)
        fichier.write("speckleRange", stereo.etendu_speck)
        fichier.write("speckleSize", stereo.taille_speck)


def sauver_donnees_calibration(cam_calib1, cam_calib2, sys3d):
    d = datetime.datetime.now()
    if sys3d is  None:
        if cam_calib1 is  None:
            cam_calib = cam_calib2
        else:
            cam_calib = cam_calib1
        cam_calib.fic_donnees = "Echiquier_" + str(cam_calib.idx_usb) +\
           "_" + str(d)+".yml"
        cam_calib.fic_donnees = cam_calib.fic_donnees.replace(':', '_')
        cam_calib.fic_donnees = cam_calib.fic_donnees.replace('-', '_')
        fichier = cv.FileStorage(cam_calib.fic_donnees, cv.FILE_STORAGE_WRITE)
        if not fichier.isOpened():
            fichier.release()
        fichier.write("nbGrille", len(cam_calib.pts_camera))
        nb_pts = 0
        for tab_pts in cam_calib.pts_camera:
            nb_pts += tab_pts.shape[0]
        fichier.write("nbPoints", nb_pts)
        for idx, tab_pts in enumerate(cam_calib.pts_camera):
            fichier.write("Grille" + str(idx), tab_pts)
        for idx, tab_pts in enumerate(cam_calib.pts_objets):
            fichier.write("Objet" + str(idx), tab_pts)
        fichier.release()
    else:
        sys3d.fic_donnees = "EchiquierStereo_" + str(d) + ".yml"
        sys3d.fic_donnees = sys3d.fic_donnees.replace(':', '_')
        sys3d.fic_donnees = sys3d.fic_donnees.replace('-', '_')
        fichier = cv.FileStorage(sys3d.fic_donnees, cv.FILE_STORAGE_WRITE)
        if not fichier.isOpened():
            fichier.release()
        fichier.write("nbGrille", len(sys3d.pts_camera_gauche))
        for idx, tab_pts in enumerate(sys3d.pts_camera_gauche):
            fichier.write("GrilleG" + str(idx), tab_pts)
        for idx, tab_pts in enumerate(sys3d.pts_camera_droite):
            fichier.write("GrilleD" + str(idx), tab_pts)
        for idx, tab_pts in enumerate(sys3d.pts_objets):
            fichier.write("Objets" + str(idx), tab_pts)
        fichier.release()

class NuageVtk:
    def __init__(self, x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=0.0, z_max=5, max_pts=1e6):
        self.max_pts = max_pts
        self.vtkPolyData = vtk.vtkPolyData()
        self.clear_pts()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(z_min, z_max)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.nuage = None
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = y_max
        self.y_max = x_max
        self.z_max = z_max

    def add_pts(self, point, color):
        pt_id = self.vtk_pts.InsertNextPoint(point[:])
        self.vtkCells.InsertNextCell(1)
        self.vtkCells.InsertCellPoint(pt_id)
        #https://www.vtk.org/Wiki/VTK/Examples/Cxx/PolyData/ColoredPoints
        self.colors.InsertNextTuple3(color[2], color[1], color[0])
        self.vtkCells.Modified()
        self.vtk_pts.Modified()
        self.vtkDepth.Modified()
    def trace(self, img, p3):
        p = None
        z = np.reshape(p3, (p3.shape[0] * p3.shape[1], 3))
        c = np.reshape(img, (img.shape[0] * img.shape[1], 3))
        idx = (z[:, 2] >= self.z_min) & (z[:, 2] < self.z_max) &\
           (z[:, 1] >= self.y_min) & (z[:, 1] < self.y_max) &\
           (z[:, 0] >= self.x_min) & (z[:, 0] < self.x_max)
        p = z[idx, :]
        c = c[idx, :]
        for i in range(len(p)):
            couleur = c[i, :]
            self.add_pts(p[i, :], couleur)
        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(self.vtkActor)
        renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        # Begin Interaction
        renderWindow.Render()
        renderWindowInteractor.Start()
    def clear_pts(self):
        self.vtk_pts = vtk.vtkPoints()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtk_pts)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        self.vtkPolyData.GetPointData().SetScalars(self.colors)


class GestionFlux(threading.Thread):
    def __init__(self, fluxvideo, idx):
        threading.Thread.__init__(self)
        self.video = fluxvideo
        self.pgm_fin = False
        self.idx = idx
        self.derniere_image = []
        self.cmd = 0
        self.verrou = threading.Lock()
        self.image_lue = True

    def run(self):
        if self.video is None:
            print("Le flux n est pas ouvert. Fin du thread")
            return
        cmd = 0
        while True:
            if self.pgm_fin:
                break
            ret, img = self.video.read()
            if ret:
                while not self.image_lue and not self.pgm_fin:
                    time.sleep(0.001)

                with self.verrou:
                    self.derniere_image = img
                    self.image_lue = False
                    cmd = self.cmd
                    self.cmd = 0
            if cmd > 0:
                if 32 <= cmd < 256:
                    cmd = chr(cmd)
                    self.gestion_cmd_camera(cmd)
            if cmd != 0:
                cmd = 0


    def get_image(self):
        if self.image_lue:
            return None
        return self.derniere_image
    def prop_camera(self, param):
        valeur = self.video.get(param[0]) + param[1]
        self.video.set(param[0], valeur)

    def gestion_cmd_camera(self, cmd):
        modif_param_camera = {
            'G': (cv.CAP_PROP_GAIN, 1, 'CAP_PROP_GAIN'),
            'g': (cv.CAP_PROP_GAIN, -1, 'CAP_PROP_GAIN'),
            'B': (cv.CAP_PROP_BRIGHTNESS, 1, 'CAP_PROP_BRIGHTNESS'),
            'b': (cv.CAP_PROP_BRIGHTNESS, -1, 'CAP_PROP_BRIGHTNESS'),
            'E': (cv.CAP_PROP_EXPOSURE, 1, 'CAP_PROP_EXPOSURE'),
            'e': (cv.CAP_PROP_EXPOSURE, -1, 'CAP_PROP_EXPOSURE'),
            'C': (cv.CAP_PROP_SATURATION, 1, 'CAP_PROP_SATURATION'),
            'c': (cv.CAP_PROP_SATURATION, -1, 'CAP_PROP_SATURATION'),
            'w': (cv.CAP_PROP_SETTINGS, 0.1, 'CAP_PROP_SETTINGS')
            }
        if cmd in modif_param_camera:
            self.prop_camera(modif_param_camera[cmd])
        else:
            print("Code inconnu", cmd)

class SuiviDistance:
    def __init__(self):
        self.stereo = CalibrageCamera3D()
        self.disparite = None
        self.m = None
        self.p = None
        self.zoom = 1
    def mesure_distance(self, event, x, y, flags, data):
        if self.disparite is None:
            return
        if event == cv.EVENT_FLAG_LBUTTON:
            dd = self.disparite[int(y / self.zoom), int(x / self.zoom)]
            if dd < 0:
                return
            p = np.array([[[x / self.zoom, y/ self.zoom, dd]]])
            if self.p is None:
                self.p = p
            else:
                self.p = np.concatenate((self.p, p), axis=0)
            self.m = cv.perspectiveTransform(src=self.p, m=self.stereo.Q)
            print("\n ++++++++++\n", self.p, "\n")
            for i in range(self.m.shape[0]):
                print(self.p[i, :], " = ", self.m[i, :], " --> ",
                      cv.norm(self.m[i, :]), "\n")
        if event == cv.EVENT_FLAG_RBUTTON:
            self.p = None

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

def gestion_cmd_camera(param_ecran, liste_thread):
    cmd_traitee = True
    if param_ecran.mode_affichage & MODE_REGLAGECAMERA != 0:
        if ord('0') <= param_ecran.code_touche_clavier <= ord('9'):
            idx_nouveau = param_ecran.code_touche_clavier - 48
            with liste_thread[idx_nouveau].verrou:
                liste_thread[idx_nouveau].cmd = \
                    liste_thread[idx_nouveau].cmd | (MODE_REGLAGECAMERA)
            with liste_thread[param_ecran.index_camera].verrou:
                liste_thread[param_ecran.index_camera].cmd = \
                    liste_thread[1].cmd & (~MODE_REGLAGECAMERA)
            param_ecran.index_camera = idx_nouveau
        elif param_ecran.code_touche_clavier == ord('R'):
            param_ecran.mode_affichage = param_ecran.mode_affichage & \
                (~MODE_REGLAGECAMERA)
            with liste_thread[param_ecran.index_camera].verrou:
                liste_thread[param_ecran.index_camera].cmd = \
                    liste_thread[param_ecran.index_camera].cmd & \
                    (~MODE_REGLAGECAMERA)
            print("Mode stereo")
        else:
            with liste_thread[param_ecran.index_camera].verrou:
                liste_thread[param_ecran.index_camera].cmd = \
                    liste_thread[param_ecran.index_camera].cmd | param_ecran.code_touche_clavier
    else:
        if param_ecran.code_touche_clavier == ord('R'):
            param_ecran.mode_affichage = param_ecran.mode_affichage | \
                (MODE_REGLAGECAMERA)
            with liste_thread[param_ecran.index_camera].verrou:
                liste_thread[param_ecran.index_camera].cmd = \
                    liste_thread[param_ecran.index_camera].cmd | \
                    (~MODE_REGLAGECAMERA)
            print("Mode reglage")
        elif param_ecran.code_touche_clavier == ord('+'):
            if param_ecran.zoom < 8:
                param_ecran.zoom *= 2
        elif param_ecran.code_touche_clavier == ord('-'):
            if param_ecran.zoom > 0.1:
                param_ecran.zoom /= 2
        else:
            cmd_traitee = False
    return cmd_traitee, param_ecran

def gestion_cmd_stereo(param_ecran, liste_thread, liste_cam_calib, sys3d, mire):
    if param_ecran.code_touche_clavier <= 0:
        return param_ecran, liste_thread, liste_cam_calib, sys3d
    caractere = chr(param_ecran.code_touche_clavier)
    if caractere == 'b':
        param_ecran.frame = np.zeros(
            (param_ecran.taille_globale[1], param_ecran.taille_globale[0], 3),
            np.uint8)
        for cam_calib  in liste_cam_calib:
            cam_calib.pts_objets = []
            cam_calib.fic_donnees = ""
            cam_calib.pts_camera = []
        sys3d.pts_camera_droite = []
        sys3d.pts_camera_gauche = []
        sys3d.fic_donnees = ""
        sys3d.pts_objets = []
    if caractere == 'c':
        if len(liste_cam_calib) <= param_ecran.index_camera or \
            len(liste_cam_calib[param_ecran.index_camera].pts_camera) == 0:
            print("Aucune grille pour la calibration\n")
        liste_cam_calib[param_ecran.index_camera].calibrer(tailleWebcam[param_ecran.index_camera])
        print("RMS = ", liste_cam_calib[param_ecran.index_camera].rms,
              "\n", liste_cam_calib[param_ecran.index_camera].mat_intrin,
              "\n", liste_cam_calib[param_ecran.index_camera].mat_dist, "\n")
        sauver_configuration("config.yml", mire, liste_cam_calib, sys3d, stereo)
    elif caractere == 'e':
        cv.imwrite("imgL.png", liste_images[0])
        cv.imwrite("imgR.png", liste_images[1])
        if suivi_dst.disparite is not None:
            disparite = suivi_dst.disparite.astype(np.float)/16
            fichier = cv.FileStorage("disparite.yml", cv.FILE_STORAGE_WRITE)
            if fichier.isOpened():
                fichier.write("Image", disparite)
            fichier2 = cv.FileStorage("disparitebrut.yml", cv.FILE_STORAGE_WRITE)
            if fichier2.isOpened():
                fichier2.write("Image", suivi_dst.disparite)
    elif caractere == 'D' or caractere == 'd':
        if param_ecran.index_camera != 1:
            liste_cam_calib[param_ecran.index_camera].pts_camera = []
        param_ecran.index_camera = 1
        r_dst = [tailleWebcam[1][0], 0, tailleWebcam[1][0], tailleWebcam[1][1]]
        y = np.zeros((tailleWebcam[1][1], tailleWebcam[1][0], 3), np.uint8)
        frame = param_ecran.frame

        if caractere == 'D':
            ret, y, _ = liste_cam_calib[1].analyse_charuco(
                liste_images[param_ecran.index_camera], y, mire)
        if caractere == 'd':
            ret, y = liste_cam_calib[1].analyse_grille(
                liste_images[param_ecran.index_camera], y, mire)
        if ret:
            frame[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = y +\
                frame[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]
            cv.imshow("Cameras", zoom(frame, param_ecran.zoom))
            if liste_cam_calib[param_ecran.index_camera].fic_donnees is not None:
                liste_cam_calib[param_ecran.index_camera].fic_donnees = None
        param_ecran.frame = frame
    elif caractere == 'G' or caractere == 'g':
        if param_ecran.index_camera != 0:
            liste_cam_calib[param_ecran.index_camera].pts_camera = []
        param_ecran.index_camera = 0
        if caractere == 'G':
            ret, param_ecran.frame = liste_cam_calib[0].analyse_charuco(
                liste_images[param_ecran.index_camera], param_ecran.frame, mire)
        elif caractere == 'g':
            ret, param_ecran.frame = liste_cam_calib[0].analyse_grille(
                liste_images[param_ecran.index_camera], param_ecran.frame, mire)
        if ret:
            if len(liste_cam_calib[param_ecran.index_camera].fic_donnees) is not None:
                liste_cam_calib[param_ecran.index_camera].fic_donnees = []
            cv.imshow("Cameras", zoom(param_ecran.frame, param_ecran.zoom))
    elif caractere == 'l':
        if param_ecran.mode_affichage&MODE_EPIPOLAIRE:
            param_ecran.mode_affichage = param_ecran.mode_affichage & (~MODE_EPIPOLAIRE)
        else:
            param_ecran.mode_affichage = param_ecran.mode_affichage | (MODE_EPIPOLAIRE)
        if sys3d.F is not None:
            sys3d.epipolar = cv.computeCorrespondEpilines(segment, 1, sys3d.F, )
            print(sys3d.epipolar)
    elif caractere == 'o':
        if param_ecran.mode_affichage & MODE_3D:
            param_ecran.mode_affichage = param_ecran.mode_affichage & ~MODE_3D
        else:
            param_ecran.mode_affichage = param_ecran.mode_affichage | MODE_3D
    elif caractere == 's':
        if param_ecran.index_camera != 3:
            sys3d.pts_camera_gauche = []
            sys3d.pts_camera_droite = []
            sys3d.pts_objets = []
            if sys3d.fic_donnees is not None:
                sys3d.fic_donnees = None
        param_ecran.index_camera = 3
        grille, param_ecran.frame = sys3d.analyse_grille(
                liste_images, param_ecran.frame, mire)
        if grille:
            cv.imshow("Cameras", zoom(param_ecran.frame, param_ecran.zoom))
    elif caractere == 'S':
        if param_ecran.index_camera != 3:
            sys3d.pts_camera_gauche = []
            sys3d.pts_camera_droite = []
            sys3d.pts_objets = []
            if sys3d.fic_donnees is not None:
                sys3d.fic_donnees = None
        param_ecran.index_camera = 3
        sys3d.analyse_charuco(liste_images, mire, param_ecran)
        cv.imshow("Cameras", zoom(param_ecran.frame, param_ecran.zoom))

    elif caractere == 't':
        if sys3d.R1 is not None and sys3d.R2 is not None:
            if param_ecran.algo_stereo != 0:
                param_ecran.algo_stereo = 0
            elif stereo.al_bm is not None:
                param_ecran.algo_stereo = 1
    elif caractere == 'T':
        if sys3d.R1 is not None:
            if param_ecran.algo_stereo != 0:
                param_ecran.algo_stereo = 0
            elif stereo.al_sgbm:
                param_ecran.algo_stereo = 2
    elif caractere == 'u':
        param_ecran.type_distorsion = (param_ecran.type_distorsion + 1) % 4
        print(param_ecran.type_distorsion)
        if param_ecran.type_distorsion == 0:
            sys3d.map11 = None
            sys3d.map12 = None
            sys3d.map21 = None
            sys3d.map22 = None
        if param_ecran.type_distorsion == 1:
            if liste_cam_calib[0].mat_intrin is not None:
                sys3d.map11, sys3d.map12 = cv.initUndistortRectifyMap(
                    liste_cam_calib[0].mat_intrin, liste_cam_calib[0].mat_dist,
                    None, None, tailleWebcam[0], cv.CV_16SC2)
            if liste_cam_calib[1].mat_intrin is not None:
                sys3d.map21, sys3d.map22 = cv.initUndistortRectifyMap(
                    liste_cam_calib[1].mat_intrin, liste_cam_calib[1].mat_dist,
                    None, None, tailleWebcam[1], cv.CV_16SC2)
        if param_ecran.type_distorsion == 2:
            if sys3d.R1 is not None and liste_cam_calib[0].mat_intrin is not None:
                sys3d.map11, sys3d.map12 = cv.initUndistortRectifyMap(
                    liste_cam_calib[0].mat_intrin, None,
                    sys3d.R1, sys3d.P1, tailleWebcam[0], cv.CV_16SC2)
            if sys3d.R2 is not None and liste_cam_calib[1].mat_intrin is not None:
                sys3d.map21, sys3d.map22 = cv.initUndistortRectifyMap(
                    liste_cam_calib[1].mat_intrin, None,
                    sys3d.R2, sys3d.P2, tailleWebcam[1], cv.CV_16SC2)
        if param_ecran.type_distorsion == 3:
            if sys3d.R1 is not None and liste_cam_calib[0].mat_intrin is not None:
                sys3d.map11, sys3d.map12 = cv.initUndistortRectifyMap(
                    liste_cam_calib[0].mat_intrin, liste_cam_calib[0].mat_dist,
                    sys3d.R1, sys3d.P1, tailleWebcam[0], cv.CV_16SC2)
            if sys3d.R2 is not None and liste_cam_calib[1].mat_intrin is not None:
                sys3d.map21, sys3d.map22 = cv.initUndistortRectifyMap(
                    liste_cam_calib[1].mat_intrin, liste_cam_calib[1].mat_dist,
                    sys3d.R2, sys3d.P2, tailleWebcam[1], cv.CV_16SC2)
    elif caractere == '3':
        if len(sys3d.pts_camera_gauche) != len(sys3d.pts_camera_droite) or\
            len(sys3d.pts_camera_gauche) == 0:
            print("Pas de grille coherente pour le calibrage 3D\n")
        sauver_donnees_calibration(liste_cam_calib[0], liste_cam_calib[1], sys3d)
        sys3d.m = [[], []]
        sys3d.d = [[], []]
        sys3d.m[0] = liste_cam_calib[0].mat_intrin.copy()
        sys3d.m[1] = liste_cam_calib[1].mat_intrin.copy()
        sys3d.d[0] = liste_cam_calib[0].mat_dist.copy()
        sys3d.d[1] = liste_cam_calib[1].mat_dist.copy()
        for param in sys3d.type_calib:
            resultat = cv.stereoCalibrate(
                sys3d.pts_objets,
                sys3d.pts_camera_gauche,
                sys3d.pts_camera_droite,
                sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                tailleWebcam[0], sys3d.R, sys3d.T, sys3d.E, sys3d.F,
                param,
                (cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 5000000, 1e-8))
            sys3d.rms, sys3d.m[0], sys3d.d[0],\
                sys3d.m[1], sys3d.d[1], sys3d.R,\
                sys3d.T, sys3d.E, sys3d.F = resultat
        print("Erreur quadratique =", sys3d.rms)
#        print(sys3d.erreur_droite_epipolaire() )
        print(sys3d.m[0], "\n", sys3d.d[0], "\n", sys3d.m[1], "\n", sys3d.d[1], "\n")
        print(sys3d.R, "\n", sys3d.T, "\n", sys3d.E, "\n", sys3d.F, "\n")
        if sys3d.R is not None:
            sys3d.R1, sys3d.R2, sys3d.P1, sys3d.P2, sys3d.Q, sys3d.valid1, sys3d.valid2 =\
                cv.stereoRectify(sys3d.m[0], sys3d.d[0], sys3d.m[1], sys3d.d[1],
                                 tailleWebcam[0], sys3d.R, sys3d.T,
                                 sys3d.R1, sys3d.R2, sys3d.P1,
                                 sys3d.P2, sys3d.Q,
                                 cv.CALIB_ZERO_DISPARITY, -1)
            sauver_configuration("config.yml", mire, liste_cam_calib, sys3d, stereo)
    return param_ecran, liste_thread, liste_cam_calib, sys3d

if __name__ == '__main__':
    param_ecran = ParamAffichage()

    mire = ParamMire()
    configActive, mire, liste_cam_calib, sys3d, stereo = charger_configuration("configStereo.yml")
    if len(liste_cam_calib) != 2:
        print("erreur dans le fichier de configuration\n")
        exit()
    liste_webcam_ouvertes = recherche_camera()
    if liste_webcam_ouvertes == []:
        print("Aucune camera trouvee")
        exit()
    if len(liste_webcam_ouvertes) != 2:
        print("Le nombre de camera <> 2\nFin du programme")
        exit()
    tailleWebcam = param_ecran.init_fenetre(liste_webcam_ouvertes, liste_cam_calib)
    liste_thread = install_threads(liste_webcam_ouvertes)


    if sys3d.R1 is not None:
        sys3d.map11, sys3d.map12 = cv.initUndistortRectifyMap(
            sys3d.m[0], sys3d.d[0], sys3d.R1,
            sys3d.P1, tailleWebcam[0], cv.CV_16SC2)
    if sys3d.R2 is not None:
        sys3d.map21, sys3d.map22 = cv.initUndistortRectifyMap(
            sys3d.m[1], sys3d.d[1], sys3d.R2,
            sys3d.P2, tailleWebcam[1], cv.CV_16SC2)
    cv.imshow("Cameras", cv.resize(src=param_ecran.frame, dsize=None,
                                   fx=param_ecran.zoom,
                                   fy=param_ecran.zoom))
    cv.namedWindow("Control", cv.WINDOW_NORMAL)
    if sys3d.map11 is not None and sys3d.map21 is not None:
        stereo.al_bm = cv.StereoBM_create(16*stereo.nb_disparite, 2 * stereo.taille_bloc + 1)
        stereo.al_sgbm = cv.StereoSGBM_create(0, 16*stereo.nb_disparite, 2 * stereo.taille_bloc + 1)
        stereo.al_bm.setPreFilterType(stereo.pre_filtre)
        stereo.al_bm.setUniquenessRatio(stereo.unicite)
        stereo.al_sgbm.setUniquenessRatio(stereo.unicite)
        stereo.al_bm.setSpeckleWindowSize(stereo.taille_speck)
        stereo.al_sgbm.setSpeckleWindowSize(stereo.taille_speck)
        stereo.al_bm.setSpeckleRange(stereo.etendu_speck)
        stereo.al_sgbm.setSpeckleRange(stereo.etendu_speck)
        ajouter_glissiere("Bloc", "Control", 2, 100,
                            stereo.taille_bloc, stereo.maj_stereo_bloc)
        ajouter_glissiere("nb_disparite", "Control", 1, 100,
                            stereo.nb_disparite,
                            stereo.maj_stereo_disp)
        ajouter_glissiere("Unicite", "Control", 3, 100,
                            stereo.unicite,
                            stereo.maj_stereo_unicite)
        ajouter_glissiere("etendu_speck", "Control", 1, 10,
                            stereo.etendu_speck,
                            stereo.maj_stereo_etendu_speck)
        ajouter_glissiere("taille_speck", "Control", 3, 100,
                            stereo.taille_speck,
                            stereo.maj_stereo_taille_speck)

        alg = cv.STEREO_SGBM_MODE_SGBM
        stereo.al_sgbm.setMode(alg)
    liste_cam_calib[0].pts_grille = np.zeros((mire.nb_lig*mire.nb_col, 1, 3),
                                             np.float32)
    liste_cam_calib[0].pts_grille[:, 0, :2] =\
        np.mgrid[0:mire.nb_col, 0:mire.nb_lig].T.reshape(-1, 2) * mire.dim_carre
    liste_cam_calib[1].pts_grille = np.zeros((mire.nb_lig * mire.nb_col, 3), np.float32)
    liste_cam_calib[1].pts_grille[:, :2] =\
         np.mgrid[0:mire.nb_col, 0:mire.nb_lig].T.reshape(-1, 2) * mire.dim_carre
    sys3d.pts_grille = np.zeros((mire.nb_lig * mire.nb_col, 3),
                                np.float32)
    sys3d.pts_grille[:, :2] = \
        np.mgrid[0:mire.nb_col, 0:mire.nb_lig].T.reshape(-1, 2) * mire.dim_carre

    suivi_dst = SuiviDistance()
    suivi_dst.stereo = sys3d
    suivi_dst.zoom = param_ecran.zoom
    cv.namedWindow("Webcam 0")
    cv.namedWindow("disparite")
    cv.setMouseCallback("Webcam 0", suivi_dst.mesure_distance)
    cv.setMouseCallback("disparite", suivi_dst.mesure_distance)
    modeReglageCam = False

    segment  = []
    for idx in range(10):
        segment.append((tailleWebcam[0][1]/2, tailleWebcam[0][0] * idx / 10.0))
    segment = np.asarray(segment)

    while param_ecran.code_touche_clavier != CODE_TOUCHE_FIN:
        param_ecran.code_touche_clavier = cv.waitKey(10)
        cmd_traitee = False
        erreur_lecture = False
        if param_ecran.code_touche_clavier > 0:
            cmd_traitee, param_ecran = gestion_cmd_camera(
                param_ecran,
                liste_thread
                )
        liste_images = lire_images(liste_thread)
        if len(liste_images) != 2 or  liste_images[0] is None or liste_images[1] is None:
            print("Probleme de lecture")
            cmd_traitee = True
            erreur_lecture = True
        if not cmd_traitee:
            param_ecran, liste_thread, liste_cam_calib, sys3d = gestion_cmd_stereo(
                param_ecran, liste_thread, liste_cam_calib, sys3d, mire)
        if not erreur_lecture and param_ecran.mode_affichage:
            if param_ecran.mode_affichage&MODE_EPIPOLAIRE and sys3d.epipolar is not None:
                for i in range(sys3d.epipolar.shape[0]):
                    cv.circle(liste_images[0], tuple(segment[i].astype(np.uint32)),
                              5, (255, 255, 0))
                    cv.line(liste_images[0], (0, int(segment[i][1])),
                            (liste_images[0].shape[1] - 1,
                             int(segment[i][1])), (255, 255, 0))
                    a = sys3d.epipolar[i][0, 0]
                    b = sys3d.epipolar[i][0, 1]
                    c = sys3d.epipolar[i][0, 2]
                    if a != 0:
                        x0 = int(-c / a)
                        x1 = int((-b * liste_images[1].shape[0] - c) / a)
                        xOrig = [(x0, 0), (x1, liste_images[1].shape[0])]
                        cv.line(liste_images[1], xOrig[0], xOrig[1], (255, 255, 0))
            if param_ecran.type_distorsion != 0 and sys3d.map11 is not None:
                liste_images[0] = cv.remap(
                    liste_images[0], sys3d.map11,
                    sys3d.map12, cv.INTER_LINEAR)
            if param_ecran.type_distorsion != 0 and sys3d.map21 is not None:
                liste_images[1] = cv.remap(
                    liste_images[1], sys3d.map21,
                    sys3d.map22, cv.INTER_LINEAR)

            for idx, image in enumerate(liste_images):
                if image is not None:
                    cv.imshow("Webcam " + str(idx),
                              zoom(image, param_ecran.zoom))
            if param_ecran.mode_affichage&MODE_EPIPOLAIRE:
                r_dst = [0, 0, 0, 0]
                for idx, image in enumerate(liste_images):
                    if idx == 0:
                        r_dst = [0, 0, tailleWebcam[idx][0], tailleWebcam[idx][1]]
                    elif idx % 2 == 1:
                        r_dst[0] += tailleWebcam[idx][0]
                    else:
                        r_dst[1] += tailleWebcam[idx][1]
                    param_ecran.frame[r_dst[1]:r_dst[1] + r_dst[3],
                                      r_dst[0]:r_dst[0] + r_dst[2]] = image
                cv.imshow("Cameras", zoom(param_ecran.frame, param_ecran.zoom))
            if param_ecran.algo_stereo:
                if param_ecran.algo_stereo == 2:
                    suivi_dst.disparite = stereo.al_sgbm.compute(liste_images[0], liste_images[1])
                else:
                    img_gauche = cv.cvtColor(liste_images[0], cv.COLOR_BGR2GRAY)
                    img_droite = cv.cvtColor(liste_images[1], cv.COLOR_BGR2GRAY)
                    suivi_dst.disparite = stereo.al_bm.compute(img_gauche, img_droite)
                suivi_dst.disparite = suivi_dst.disparite / 16
                disp8 = suivi_dst.disparite.astype(np.uint8)
                disp8cc = cv.applyColorMap(disp8, cv.COLORMAP_JET)
                cv.imshow("disparite", zoom(disp8cc, param_ecran.zoom))
                if suivi_dst.disparite is not None and (param_ecran.mode_affichage&MODE_3D):
                    disparite = suivi_dst.disparite.astype(np.float32)
                    xyz = cv.reprojectImageTo3D(disparite, sys3d.Q, False)

                    nuage3d = NuageVtk()
                    nuage3d.trace(liste_images[0], xyz)
                    param_ecran.mode_affichage = param_ecran.mode_affichage & (~MODE_3D)

    for thread in liste_thread:
        with thread.verrou:
            thread.pgm_fin = True
    time.sleep(0.5)
    sauver_configuration("config.yml", mire, liste_cam_calib, sys3d, stereo)
    cv.destroyAllWindows()
    exit()

