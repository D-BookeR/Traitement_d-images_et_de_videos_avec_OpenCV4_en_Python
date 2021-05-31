import time
import threading
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
MODE_AFFICHAGE = np.int(0x100)
MODE_REGLAGECAMERA = np.int(0x1000)
NBCAMERA = 4
CAMAPI = cv.CAP_DSHOW

class ParamAffichage:
    def __init__(self):
        self.pano_actif = False
        self.index_camera = 0
        self.mode_affichage = MODE_AFFICHAGE
        self.frame = []
        self.zoom = 1
        self.code_touche_clavier = 0
        self.taille_globale = []
        self.dct_surface = {
            's':"stereographic",
            'S':"spherical",
            'f':"fisheye",
            'p':"plane",
            'A':'affine',
            'y':"cylindrical"
            }

class ParamPano:
    def __init__(self):
        self.liste_pos_coin = []
        self.init = False
        self.nb_point_cle = 500
        self.seuil_appariement = 0.3
        self.seuil_confiance = 1
        self.indices = []
        self.matcher_type = "Best"
        self.type_estimateur = "homography"
        self.fct_couture = "dp_color"
        self.surface_compo = "plane"
        self.correction_horizon = False
        self.try_cuda = False
        self.correction_exposition = cv.detail.ExposureCompensator_GAIN_BLOCKS
        self.force_melange = 100
        self.couture_visible = False
        self.liste_taille_masque = []
        self.liste_masque_compo = []
        self.liste_masque_compo_piece = []
        self.liste_masque_piece_init = False
#        self.liste_masque_piece = []
        self.cameras = []
        self.focales = []
        self.parametre_ba = ""
        self.focale_moyenne = 1
        self.algo_composition = None
        self.algo_correct_expo = None
        self.composition = None
        self.gains = []
        self.liste_fct_cout = ['ray', 'reproj', 'affine', 'no']
        self.fct_cout = self.liste_fct_cout[0]
        self.cout = {
            self.liste_fct_cout[0]:cv.detail_BundleAdjusterRay,
            self.liste_fct_cout[1]:cv.detail_BundleAdjusterReproj,
            self.liste_fct_cout[2]:cv.detail_BundleAdjusterAffine,
            self.liste_fct_cout[3]:cv.detail_NoBundleAdjuster
            }
        self.liste_couture = ["no", "voronoi",
                        "gc_color", "gc_colorgrad",
                        "dp_color", "dp_colorgrad"
                        ]
        self.fct_couture = self.liste_couture[0]
        self.couture = {
            self.liste_couture[0]: cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO),
            self.liste_couture[1]: cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM),
            self.liste_couture[2]: cv.detail_GraphCutSeamFinder("COST_COLOR"),
            self.liste_couture[3]: cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD"),
            self.liste_couture[4]: cv.detail_DpSeamFinder("COLOR"),
            self.liste_couture[5]: cv.detail_DpSeamFinder("COLOR_GRAD")
            }

    def remise_a_zero(self):
        self.liste_taille_masque = []
        self.liste_pos_coin = []
        self.liste_masque_compo = []
        self.liste_masque_compo_piece = []
#        self.liste_masque_piece = []
        self.liste_masque_piece_init = False
        self.init = False
        self.cameras = []
        self.focales = []
        self.gains = []

    def print(self):
        print ("Appariement : ", self.matcher_type)
        print ("Transformation : ", self.type_estimateur)
        print ("Couture : ", self.fct_couture)
        print ("Surface : ", self.surface_compo)
        print ("Exposition : ", self.correction_exposition)
        print ("Ajustement : ", self.fct_cout)

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


def sauver_configuration(pano):
    fic_config = cv.FileStorage("WebCamPanoramique.yml", cv.FILE_STORAGE_WRITE)
    fic_config.write("init", 1)
    fic_config.write("fct_couture", pano.fct_couture)
    fic_config.write("surface_composition", pano.surface_compo)
    fic_config.write("seuil_confiance", pano.seuil_confiance)
    fic_config.write("seuil_appariement", pano.seuil_appariement)
    fic_config.write("correction_exposition", pano.correction_exposition)
    fic_config.write("force_melange", pano.force_melange)
    fic_config.write("fct_cout", pano.fct_cout)
    fic_config.write("taille", len(pano.liste_pos_coin))
    fic_config.write("focale_moyenne", pano.focale_moyenne)
    fic_config.write("indices", pano.indices)
    pano.gains = pano.algo_correct_expo.getMatGains(None)
    fic_config.write("gainsize", len(pano.gains))
    for i in range(len(pano.gains)):
        fic_config.write("gain" + str(i), pano.gains[i])
    for i in range(len(pano.liste_pos_coin)):
        fic_config.write("focal" + str(i), pano.focales[i])
        fic_config.write("indice" + str(i), np.array(pano.indices[i]))
        fic_config.write("coin" + str(i), np.array(pano.liste_pos_coin[i], dtype=np.int))
        fic_config.write("liste_taille_masque" + str(i),
                         np.array(pano.liste_taille_masque[i], np.int))
        fic_config.write("masque" + str(i), pano.liste_masque_compo[i].get())
        fic_config.write("cameraRot" + str(i), pano.cameras[i].R)
        fic_config.write("cameraFocal" + str(i), pano.cameras[i].focal)
        fic_config.write("cameraPPX" + str(i), pano.cameras[i].ppx)
        fic_config.write("cameraPPY" + str(i), pano.cameras[i].ppy)

def restaurer_configuration():
    try:
        fic_config = cv.FileStorage("WebCamPanoramique.yml", cv.FILE_STORAGE_READ)
    except IOError:
        print("file is empty or does not exist")
        return False, ParamPano()
    except (cv.error,SystemError):
        print("invalid file")
        return False, ParamPano()
    if not fic_config.isOpened():
        return False, ParamPano()
    pano = ParamPano()
    pano.init = fic_config.getNode("init")
    pano.fct_couture = fic_config.getNode("fct_couture").string()
    pano.surface_compo = fic_config.getNode("surface_composition").string()
    pano.fct_cout = fic_config.getNode("fct_cout").string()
    pano.correction_exposition = int(fic_config.getNode("correction_exposition").real())
    pano.force_melange = fic_config.getNode("force_melange").real()
    pano.seuil_confiance = fic_config.getNode("seuil_confiance").real()
    pano.seuil_appariement = fic_config.getNode("seuil_appariement").real()
    taille = int(fic_config.getNode("taille").real())
    pano.focale_moyenne = fic_config.getNode("focale_moyenne").real()
    pano.indices = fic_config.getNode("indices").mat()
    for i in range(taille):
        taille = fic_config.getNode("tailleImage" + str(i)).mat()
        focale = fic_config.getNode("focal" + str(i)).real()
        pano.focales.append(focale)
        pos_coin = fic_config.getNode("coin" + str(i)).mat()
        pos_coin = pos_coin.astype(int)
        pano.liste_pos_coin.append(tuple(pos_coin.transpose()[0].tolist()))
        taille_masque = fic_config.getNode("liste_taille_masque" + str(i)).mat()
        taille_masque = taille_masque.astype(np.uint32)
        pano.liste_taille_masque.append(tuple(taille_masque.transpose()[0].tolist()))
        img_masque = fic_config.getNode("masque" + str(i)).mat()
        pano.liste_masque_compo.append(cv.UMat(img_masque))
        camera = cv.detail_CameraParams()
        camera.aspect = 1
        camera.t = np.zeros((3, 1), np.float64)
        camera.R = fic_config.getNode("cameraRot" + str(i)).mat()
        camera.focal = fic_config.getNode("cameraFocal" + str(i)).real()
        camera.ppx = fic_config.getNode("cameraPPX" + str(i)).real()
        camera.ppy = fic_config.getNode("cameraPPY" + str(i)).real()
        pano.cameras.append(camera)
    taille = int(fic_config.getNode("gainsize").real())
    pano.gains = []
    for i in range(taille):
        node = fic_config.getNode("gain" + str(i))
        if node.isReal() or node.isInt():
            img_gain = np.array([node.real()])
        else:
            img_gain = node.mat()
        pano.gains.append(img_gain)
    pano.composition = cv.PyRotationWarper(pano.surface_compo, pano.focale_moyenne)
    pano.algo_correct_expo = cv.detail.ExposureCompensator_createDefault(pano.correction_exposition)
    pano.algo_correct_expo.setMatGains(pano.gains)
    pano.algo_correct_expo.setUpdateGain(False)
    return True, pano


def init_panorama(liste_images, pano):
    nb_images = len(liste_images)
    if nb_images < 2:
        print("Il faut au moins 2 images")
        return False, pano
    algo_descripteur = cv.ORB.create(pano.nb_point_cle)
    descripteurs = cv.detail.computeImageFeatures(algo_descripteur, liste_images)
    if pano.matcher_type == "affine":
        algo_apparier = cv.detail_AffineBestOf2NearestMatcher(
            True,
            pano.try_cuda,
            pano.seuil_appariement)
    else:
        algo_apparier = cv.detail_AffineBestOf2NearestMatcher(
            False,
            pano.try_cuda,
            pano.seuil_appariement)
    appariement_image = algo_apparier.apply2(descripteurs)
    pano.indices = cv.detail.leaveBiggestComponent(
        descripteurs,
        appariement_image,
        pano.seuil_confiance
        )
    nb_images = len(pano.indices)
    if nb_images < 2:
        print("Echec de l'appariement")
        return False, pano
    pano.remise_a_zero()
    if pano.type_estimateur == "affine":
        estimateur = cv.detail_AffineBasedEstimator()
    else:
        estimateur = cv.detail_HomographyBasedEstimator()
    ret, pano.cameras = estimateur.apply(descripteurs, appariement_image, None)
    if not ret:
        print("Echec de l'estimation des modeles.")
        return False, pano
    for cam in pano.cameras:
        cam.R = cam.R.astype(np.float32)
    if pano.fct_cout not in pano.cout:
        print("Fonction de cout inconnue: ", pano.fct_cout)
        return False, pano
    ajuster = pano.cout[pano.fct_cout]()
    ajuster.setConfThresh(pano.seuil_confiance)
    refine_mask = np.ones((3, 3), np.uint8)
    ajuster.setRefinementMask(refine_mask)

    ret, pano.cameras = ajuster.apply(descripteurs, appariement_image, pano.cameras)
    if not ret:
        print("Echec de l'ajustement des parametres.")
        return False, pano
    for cam in pano.cameras:
        pano.focales.append(cam.focal)
    sorted(pano.focales)

    if len(pano.focales) % 2 == 1:
        pano.focale_moyenne = pano.focales[len(pano.focales) // 2]
    else:
        pano.focale_moyenne = (pano.focales[len(pano.focales) // 2] +
                               pano.focales[len(pano.focales) // 2 - 1]) / 2
    images_projetees = []
    images_projetees_float = []

    pano.composition = cv.PyRotationWarper(pano.surface_compo, pano.focale_moyenne)
    for i in range(nb_images):
        idx = pano.indices[i][0]
        cam_int = pano.cameras[idx].K().astype(np.float32)
        coins, image_wp = pano.composition.warp(
            liste_images[idx],
            cam_int,
            pano.cameras[idx].R,
            cv.INTER_LINEAR,
            cv.BORDER_REFLECT
            )
        pano.liste_pos_coin.append(coins)
        pano.liste_taille_masque.append((image_wp.shape[1], image_wp.shape[0]))
        images_projetees.append(image_wp)
        umat = cv.UMat(255 *
                       np.ones((liste_images[pano.indices[i][0]].shape[0],
                                liste_images[pano.indices[i][0]].shape[1]),
                               np.uint8)
                       )

        _, mask_wp = pano.composition.warp(umat, cam_int,
                                           pano.cameras[idx].R,
                                           cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        pano.liste_masque_compo.append(mask_wp)

    for img in images_projetees:
        imgf = img.astype(np.float32)
        images_projetees_float.append(imgf)
    pano.algo_correct_expo = cv.detail.ExposureCompensator_createDefault(pano.correction_exposition)
    pano.algo_correct_expo.feed(pano.liste_pos_coin, images_projetees, pano.liste_masque_compo)

    if pano.fct_couture not in pano.liste_couture:
        print("type de couture inconnue :", pano.fct_couture)
        return False, pano
    pano.couture[pano.fct_couture].find(
        images_projetees_float,
        pano.liste_pos_coin,
        pano.liste_masque_compo
        )
    if len(pano.indices) == len(liste_images):
        sauver_configuration(pano)
    return True, pano



def composer_panorama(liste_images, pano):
    nb_images = pano.indices.shape[0]
    melangeur = None
    for idx  in range(nb_images):
        index_image = int(pano.indices[idx])
        cam_int = pano.cameras[index_image].K().astype(np.float32)
        _, img_projetee = pano.composition.warp(
            liste_images[index_image],
            cam_int,
            pano.cameras[index_image].R,
            cv.INTER_LINEAR, cv.BORDER_REFLECT
            )
        pano.algo_correct_expo.apply(
            idx,
            pano.liste_pos_coin[idx],
            img_projetee,
            pano.liste_masque_compo[idx]
            )

        if not pano.liste_masque_piece_init:
            if pano.couture_visible:
                dilated_mask = cv.erode(pano.liste_masque_compo[idx], None)
            else:
                dilated_mask = cv.dilate(pano.liste_masque_compo[idx], None)
            dilated_mask = cv.bitwise_and(dilated_mask, pano.liste_masque_compo[idx])
            pano.liste_masque_compo_piece.append(dilated_mask)
        if melangeur is None:
            rect_dst = cv.detail.resultRoi(corners=pano.liste_pos_coin,
                                           sizes=pano.liste_taille_masque)
            melangeur = cv.detail.Blender_createDefault(cv.detail.Blender_MULTI_BAND)
            melangeur.prepare(rect_dst)
        try:
            melangeur.feed(img_projetee,
                           pano.liste_masque_compo_piece[idx],
                           pano.liste_pos_coin[idx]
                           )
        except cv.error:
            pano.init =  False
            return [], pano
    pano.liste_masque_piece_init = True
    result, _ = melangeur.blend(None, None)
    image_pano = np.uint8(np.clip(result, 0, 255))
    pano.init =  True
    return image_pano, pano

def gestion_cmd_camera(param_ecran, liste_thread):
    cmd_traitee =  True
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
            print("Mode pano")
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


def gestion_cmd_clavier(param_ecran, pano_cameras, liste_thread):
    cmd_traitee, param_ecran = gestion_cmd_camera(param_ecran, liste_thread)
    if not cmd_traitee:
        nouveau_pano = ParamPano()
        maj_pano = False
        if chr(param_ecran.code_touche_clavier) in param_ecran.dct_surface:
            nouveau_pano.surface_compo = \
                param_ecran.dct_surface[chr(param_ecran.code_touche_clavier)]
            if param_ecran.code_touche_clavier == ord('A'):
                nouveau_pano.matcher_type = "affine"
            else:
                nouveau_pano.matcher_type = "Best"
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('a'):
            param_ecran.mode_affichage = not param_ecran.mode_affichage
        elif param_ecran.code_touche_clavier == ord('r'):
            param_ecran.pano_actif = False
            pano_cameras = nouveau_pano
            param_ecran.frame = np.zeros(
                (param_ecran.taille_globale[1], param_ecran.taille_globale[0], 3),
                np.uint8)
        elif param_ecran.code_touche_clavier == ord('l'):
            pano_cameras.couture_visible = not pano_cameras.couture_visible
            pano_cameras.liste_masque_compo_piece = []
            pano_cameras.liste_masque_piece_init = False
        elif param_ecran.code_touche_clavier == ord('M'):
            nouveau_pano = pano_cameras
            nouveau_pano.seuil_confiance += 0.1
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('m'):
            nouveau_pano = pano_cameras
            if nouveau_pano.seuil_confiance > 0.1:
                nouveau_pano.seuil_confiance -= 0.1
                maj_pano = True
        elif param_ecran.code_touche_clavier == ord('O'):
            nouveau_pano = pano_cameras
            nouveau_pano.seuil_appariement += 0.1
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('o'):
            nouveau_pano = pano_cameras
            if nouveau_pano.seuil_appariement > 0.1:
                nouveau_pano.seuil_appariement -= 0.1
                maj_pano = True
        elif param_ecran.code_touche_clavier == ord('x'):
            nouveau_pano = pano_cameras
            nouveau_pano.correction_exposition = \
                (nouveau_pano.correction_exposition + 1) % \
                cv.detail.ExposureCompensator_CHANNELS_BLOCKS
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('n'):
            nouveau_pano = pano_cameras
            if nouveau_pano.nb_point_cle > 60:
                nouveau_pano.nb_point_cle -= 50
                maj_pano = True
        elif param_ecran.code_touche_clavier == ord('N'):
            nouveau_pano = pano_cameras
            nouveau_pano.nb_point_cle += 50
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('b'):
            nouveau_pano = pano_cameras
            idx = pano_cameras.liste_fct_cout.index(pano_cameras.fct_cout)
            nouveau_pano.fct_cout = pano_cameras.liste_fct_cout[
                (idx + 1) % len(pano_cameras.liste_fct_cout)]
            maj_pano = True
        elif param_ecran.code_touche_clavier == ord('c'):
            nouveau_pano = pano_cameras
            idx = pano_cameras.liste_couture.index(pano_cameras.fct_couture)
            nouveau_pano.fct_couture = pano_cameras.liste_couture[
                (idx + 1) % len(pano_cameras.liste_couture)]
            maj_pano = True
        if maj_pano:
            liste_images = lire_images(liste_thread)
            nouveau_pano.print()
            param_ecran.pano_actif, nouveau_pano = \
                init_panorama(liste_images, nouveau_pano)
            if param_ecran.pano_actif:
                pano_cameras = nouveau_pano

    return param_ecran, pano_cameras

def zoom_image(image, param_ecran):
    if param_ecran.zoom == 1:
        return image
    return cv.resize(image, None, None,
                     param_ecran.zoom ,
                     param_ecran.zoom 
                     )


if __name__ == '__main__':
    liste_webcam_ouvertes = recherche_camera()
    if liste_webcam_ouvertes == []:
        print("Aucune camera trouvee")
        exit()
    param_ecran = ParamAffichage()

    param_ecran.pano_actif, pano_cameras = restaurer_configuration()
    if not param_ecran.pano_actif or len(pano_cameras.cameras) != len(liste_webcam_ouvertes):
        pano_cameras = ParamPano()
    pano_cameras.print()
    param_ecran.taille_globale = [512, 512]

    liste_thread = install_threads(liste_webcam_ouvertes)
    if len(pano_cameras.cameras) != len(liste_webcam_ouvertes):
        param_ecran.pano_actif = False
    elif param_ecran.pano_actif:
        param_ecran.pano_actif = pano_cameras.init
    param_ecran.frame = np.zeros(
        (param_ecran.taille_globale[1], param_ecran.taille_globale[0], 3),
        np.uint8)
    param_ecran.code_touche_clavier = 0
    while param_ecran.code_touche_clavier != CODE_TOUCHE_FIN:
        param_ecran.code_touche_clavier = cv.waitKey(10)
        if param_ecran.code_touche_clavier > 0:
            param_ecran, pano_cameras = gestion_cmd_clavier(
                param_ecran,
                pano_cameras,
                liste_thread
                )
        liste_images = lire_images(liste_thread)
        if param_ecran.mode_affichage and liste_images:
            for idx, image in enumerate(liste_images):
                if  image is not None:
                    image_tmp = zoom_image(image, param_ecran)
                    cv.imshow("Webcam" + str(idx), image_tmp)
            if param_ecran.pano_actif:
                image_pano, pano_cameras = composer_panorama(liste_images, pano_cameras)
                if pano_cameras.init:
                    image_tmp = zoom_image(image_pano, param_ecran)
                    cv.imshow("Pano", image_tmp)
            else:
                image_tmp = zoom_image(param_ecran.frame, param_ecran)
                cv.imshow("Pano", image_tmp)

    for thread in liste_thread:
        with thread.verrou:
            thread.pgm_fin = True
    time.sleep(0.5)
    cv.destroyAllWindows()
