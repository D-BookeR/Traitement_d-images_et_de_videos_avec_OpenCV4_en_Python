import math
import numpy as np
import cv2 as cv

CODE_TOUCHE_ECHAP = 27
CODE_TOUCHE_FIN = CODE_TOUCHE_ECHAP
DESC_MOMENT = 0
DESC_HU = 1
DESC_FD = 2

class OutilsDessin:
    def __init__(self):
        self.feuille = None
        self.taille_crayon = 3
        self.couleur_crayon = (255, 255, 255)
        self.efface_feuille = True
    def gestion_crayon(self, evt, mouse_x, mouse_y, type, _):
        if type == cv.EVENT_FLAG_LBUTTON:
            if self.efface_feuille:
                self.feuille.fill(0)
                self.efface_feuille = False
            if mouse_x < self.feuille.shape[1] and mouse_y < self.feuille.shape[1]:
                cv.circle(self.feuille, (mouse_x, mouse_y),
                          self.taille_crayon, self.couleur_crayon, -1)

def polygone_bruite(polygone, n):
    p = polygone
    if n == 0:
        return p
    p = p + n * np.random.random_sample(p.shape) - n / 2
    ctr = np.empty(shape=(0, 2))
    min_x = p[0][0]
    max_x = p[0][0]
    min_y = p[0][1]
    max_y = p[0][1]
    for i in range(p.shape[0]):
        next = i + 1
        if next == p.shape[0]:
            next = 0
        u = p[next] - p[i]
        d = int(cv.norm(u))
        a = np.arctan2(u[1], u[0])
        step = 1
        if n != 0:
            step = d // n
        for j in range(1, int(d), int(max(step, 1))):
            while  True:
                pt_act = u * j / d
                r = n * np.random.random_sample()
                theta = a + 2 * math.pi * np.random.random_sample()
                p_new = np.array([
                    r * np.cos(theta) + pt_act[0] + p[i][0],
                    r*np.sin(theta) + pt_act[1] + p[i][1]
                    ])
                if p_new[0] >= 0 and p_new[1] >= 0:
                    break
            if p_new[0] < min_x:
                min_x = p_new[0]
            if p_new[0] > max_x:
                max_x = p_new[0]
            if p_new[1] < min_y:
                min_y = p_new[1]
            if p_new[1] > max_y:
                max_y = p_new[1]
            ctr = np.vstack((ctr, p_new))
    ctr = ctr.reshape((ctr.shape[0], 1, 2))
    contours = [ctr.astype(np.int32)]
    frame = np.zeros((int(max_y + 2), int(max_x + 2)), np.uint8)
    cv.drawContours(frame, contours, 0, 255, -1)
    ctr, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return ctr[0]

def normalise_contour(c, taille_boite):
    r = cv.boundingRect(c.astype(np.float32))
    k = 1.0/max(r[0]+r[2], r[1]+r[3])*(taille_boite-20)
    for i in range(c.shape[0]):
        c[i, 0, 0] = (c[i, 0, 0] - r[0] + 10) * k
        c[i, 0, 1] = (c[i, 0, 1] - r[1] + 10) * k
    return c.astype(np.int32)


class BaseFormes:
    def __init__(self):
        self.base_donnees = None
        self.base_donnees_ann = None
        self.formes = []
        self.nb_forme_par_niveau_bruit = 40
        self.nb_niveau_bruit = 8
        self.min_f = None
        self.max_f = None
        self.formes.append(np.array([(10, 10), (70, 10), (70, 70)]))
        self.formes.append(np.array([(150, 50), (250, 50), (250, 250),
                                     (200, 250), (100, 75)]))
        self.formes.append(np.array([(100, 80), (110, 80), (110, 100),
                                     (130, 100), (130, 110), (110, 110),
                                     (110, 130), (100, 130), (100, 110),
                                     (80, 110), (80, 100), (100, 100)]))
        self.formes.append(np.array([(300, 300), (500, 300), (600, 400), (600, 500),
                                     (500, 600), (300, 600), (200, 500), (200, 400)]))

    def donnees_modele(self):
        fic_formes = cv.FileStorage("formes.xml", cv.FILE_STORAGE_READ)
        if fic_formes.isOpened() is False or  fic_formes.getNode("PolygoneRef").empty():
            fic_formes.release()
            fic_formes = cv.FileStorage("formes.xml", cv.FILE_STORAGE_WRITE)
            fic_formes.write("nbPoly", len(self.formes))
            for idx, forme in enumerate(self.formes):
                fic_formes.write("poly" + str(idx), forme)
        else:
            nb_forme = fic_formes.getNode("nbPoly").int()
            for idx in range(nb_forme):
                forme = fic_formes.getNode("poly" + str(idx)).mat()
                self.formes.append(forme)
            fic_formes.release()
        if self.nb_forme_par_niveau_bruit == 0:
            return
        nb_ligne = int(len(self.formes) * self.nb_forme_par_niveau_bruit * self.nb_niveau_bruit)
        liste_desc = []
        responses_ann = np.zeros((nb_ligne, len(self.formes)), np.float32)
        responses = np.zeros((nb_ligne, 1), np.float32)
        for idx, forme in enumerate(self.formes):
            offset_row = idx * self.nb_forme_par_niveau_bruit * self.nb_niveau_bruit
            noise_level = 1
            for i in range(self.nb_niveau_bruit):
                for j in range(self.nb_forme_par_niveau_bruit):
                    c = polygone_bruite(self.formes[idx], noise_level)
                    desc = descripteur_contour(c)
                    liste_desc.append(desc)
                    responses_ann[i * self.nb_forme_par_niveau_bruit + j + offset_row, idx] = 1
                    responses[i * self.nb_forme_par_niveau_bruit + j + offset_row, 0] = idx
                    if j % 10 == 0:
                        frame = np.zeros((700, 700), np.uint8)
                        contour = []
                        c = normalise_contour(c, 100)
                        contour.append(c)
                        cv.drawContours(frame, contour, len(contour) - 1, (255), -1)
                        cv.imshow("Contour Ref.", frame)
                        cv.waitKey(10)
                noise_level += 1
        tab_desc = np.array(liste_desc).reshape((len(liste_desc), liste_desc[0].shape[1]))
        if self.min_f is None:
            self.min_f = tab_desc.min(axis=0).reshape((1, liste_desc[0].shape[1]))
            self.max_f = tab_desc.max(axis=0).reshape((1, liste_desc[0].shape[1]))
        tab_desc = (tab_desc - self.min_f) / (self.max_f - self.min_f)
        type_variable = np.zeros((tab_desc.shape[1] + 1, 1), np.uint8)
        type_variable_ann = np.zeros((tab_desc.shape[1] + len(self.formes), 1), np.uint8)
        for idx in range(tab_desc.shape[1]):
            type_variable[i, 0] = cv.ml.VAR_NUMERICAL
            type_variable_ann[i, 0] = cv.ml.VAR_NUMERICAL
        for idx, _ in enumerate(self.formes):
            type_variable_ann[tab_desc.shape[1] +idx, 0] = cv.ml.VAR_NUMERICAL
        type_variable[tab_desc.shape[1], 0] = cv.ml.VAR_CATEGORICAL

        self.base_donnees = cv.ml.TrainData_create(tab_desc, cv.ml.ROW_SAMPLE, responses,
                                                   None, None, None, type_variable)
        self.base_donnees_ann = cv.ml.TrainData_create(tab_desc, cv.ml.ROW_SAMPLE, responses_ann,
                                                       None, None, None, type_variable_ann)
        self.base_donnees.shuffleTrainTest()
        self.base_donnees_ann.shuffleTrainTest()
        self.base_donnees_ann.setTrainTestSplitRatio(0.8)
        self.base_donnees.setTrainTestSplitRatio(0.8)
        cv.destroyWindow("Contour Ref.")

def descripteur_contour(contour, type_desc=DESC_FD):
    descripteur = []
    m = cv.moments(contour)
    if type_desc == DESC_HU:
        descripteur = np.zeros((1, 7), np.float32)
        hu = cv.HuMoments(m)
        for k in range(7):
            descripteur[0, k] = hu[k]
    elif type_desc == DESC_MOMENT:
        peri = cv.arcLength(contour, true)
        descripteur = np.zeros((1, 8), np.float32)
        descripteur[0, 0] = m.nu20
        descripteur[0, 1] = m.nu11
        descripteur[0, 2] = m.nu02
        descripteur[0, 3] = m.nu30
        descripteur[0, 4] = m.nu21
        descripteur[0, 5] = m.nu12
        descripteur[0, 6] = m.nu03
        descripteur[0, 7] = m.m00 / peri/ peri
    elif type_desc == DESC_FD:
        descripteur = cv.ximgproc.fourierDescriptor(contour, nbElt=256, nbFD=16)
        descripteur = descripteur.reshape((1, descripteur.shape[0] * descripteur.shape[2]))
        descripteur = descripteur.astype(np.float32)
    return descripteur

def affiche_erreur_modele(machine, base_donnees):
    print("Modele ", machine.getDefaultName())
    retval, _ = machine.calcError(base_donnees, True, None)
    print("Erreur sur les donnees tests : ", retval)
    retval, _ = machine.calcError(base_donnees, False, None)
    print("Erreur sur les donnees d'entrainement : ", retval)

def init_svm(mes_donnees):
    try:
        machine = cv.ml.SVM_load("opencv_ml_svm.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.SVM_create()
        machine.setKernel(cv.ml.SVM_RBF)
        machine.setType(cv.ml.SVM_C_SVC)
        machine.setGamma(1)
        machine.setC(1)
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine


def init_ann(mes_donnees):
    try:
        machine = cv.ml.ANN_MLP_load("opencv_ml_ann_mlp.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if len(mes_donnees.formes) == 0:
            mes_donnees.donnees_modele()
        machine = cv.ml.ANN_MLP_create()
        layer_sizes = np.zeros((3), np.int32)
        layer_sizes[0] = mes_donnees.base_donnees_ann.getNVars()
        layer_sizes[1] = mes_donnees.base_donnees_ann.getNVars()
        layer_sizes[2] = len(mes_donnees.formes)
        machine.setLayerSizes(layer_sizes)
        machine.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM)
        machine.setTrainMethod(cv.ml.ANN_MLP_RPROP)
        machine.setTermCriteria((cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 5000000, 1e-8))
        machine.train(mes_donnees.base_donnees_ann)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees_ann)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine

def init_normal_bayes(mes_donnees):
    try:
        machine = cv.ml.NormalBayesClassifier_load("opencv_ml_nbayes.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.NormalBayesClassifier_create()
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine

def init_knearest(mes_donnees):
    try:
        machine = cv.ml.KNearest_load("opencv_ml_knn.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.KNearest_create()
        machine.setAlgorithmType(cv.ml.KNearest_BRUTE_FORCE)
        machine.setIsClassifier(True)
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine

def init_logistic_regression(mes_donnees):
    try:
        machine = cv.ml.LogisticRegression_load("opencv_ml_lr.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.LogisticRegression_create()
        machine.setLearningRate(0.1)
        machine.setIterations(1000)
        machine.setRegularization(cv.ml.LogisticRegression_REG_L2)
        machine.setTrainMethod(cv.ml.LogisticRegression_BATCH)
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine

def init_em(mes_donnees):
    try:
        machine = cv.ml.EM_load("opencv_ml_em.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.EM_create()
        machine.setClustersNumber(len(mes_donnees.formes))
        machine.setCovarianceMatrixType(cv.ml.EM_COV_MAT_DIAGONAL)
        machine.setTermCriteria((cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 1000, 0.1))
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine

def init_rtrees(mes_donnees):
    try:
        machine = cv.ml.RTrees_load("opencv_ml_rtrees.xml")
        if mes_donnees.min_f is None:
            fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_READ)
            node = fichier.getNode("minF")
            if node.empty() is False:
                mes_donnees.min_f = node.mat()
            mes_donnees.max_f = fichier.getNode("maxF").mat()
        return machine
    except (cv.error, SystemError, IOError):
        if mes_donnees.base_donnees is None:
            mes_donnees.donnees_modele()
        machine = cv.ml.RTrees_create()
        machine.train(mes_donnees.base_donnees)
        if machine.isTrained():
            affiche_erreur_modele(machine, mes_donnees.base_donnees)
        else:
            print("ERREUR lors de l'initialisation du modèle")
            return machine
        machine.save(machine.getDefaultName() + ".xml")
        fichier = cv.FileStorage(machine.getDefaultName() + ".xml", cv.FILE_STORAGE_APPEND)
        fichier.write("minF", mes_donnees.min_f)
        fichier.write("maxF", mes_donnees.max_f)
        fichier.release()
        return machine


if __name__ == '__main__':
    modele_stat = []
    mes_formes = BaseFormes()
    modele_svm = init_svm(mes_formes)
    if modele_svm and modele_svm.isTrained():
        modele_stat.append(modele_svm)
    modele_ann = init_ann(mes_formes)
    if modele_ann and modele_ann.isTrained():
        modele_stat.append(modele_ann)
    modele_normal_bayes = init_normal_bayes(mes_formes)
    if modele_normal_bayes and modele_normal_bayes.isTrained():
        modele_stat.append(modele_normal_bayes)
#    modele_knearest = init_knearest(mes_formes)
#    if modele_knearest and modele_knearest.isTrained():
#        modele_stat.append(modele_knearest)
    modele_logistic_regression = init_logistic_regression(mes_formes)
    if modele_logistic_regression and modele_logistic_regression.isTrained():
        modele_stat.append(modele_logistic_regression)
    modele_rtrees = init_rtrees(mes_formes)
    if modele_rtrees and modele_rtrees.isTrained():
        modele_stat.append(modele_rtrees)
    modele_em = init_em(mes_formes)
    if modele_em and modele_em.isTrained():
        modele_stat.append(modele_em)

    code = 0
    nom_fenetre = "Feuille"
    cv.namedWindow(nom_fenetre)
    abcdere = OutilsDessin()
    nom_fenetre = "Feuille"
    abcdere.feuille = np.zeros((700, 700), np.uint8)
    cv.setMouseCallback(nom_fenetre, abcdere.gestion_crayon)
    while code != CODE_TOUCHE_FIN:
        cv.imshow(nom_fenetre, abcdere.feuille)
        code = cv.waitKey(40)
        if ord('0') <= code < ord('0') + len(mes_formes.formes):
            abcdere.feuille.fill(0)
            if code - ord('0') < len(mes_formes.formes):
                abcdere.feuille = cv.drawContours(abcdere.feuille, mes_formes.formes,
                                                  code - 48, abcdere.couleur_crayon,
                                                  abcdere.taille_crayon)
        if code == ord('e'):
            abcdere.feuille.fill(0)
            cv.destroyWindow("formes retenues")
            cv.destroyWindow("Classement")
        elif code == ord('+'):
            abcdere.taille_crayon += 1
            if abcdere.taille_crayon > 10:
                abcdere.taille_crayon = 10
        elif code == ord('-'):
            abcdere.taille_crayon -= 1
            if abcdere.taille_crayon < 1:
                abcdere.taille_crayon = 1
        elif code == ord('c'):
            contours, arbreContour = cv.findContours(abcdere.feuille,
                                                     cv.RETR_EXTERNAL,
                                                     cv.CHAIN_APPROX_NONE)
            forme_retenue = np.zeros(abcdere.feuille.shape, np.uint8)

            cv.waitKey(10)
            if len(contours) == 1:
                desc = descripteur_contour(contours[0])
                desc = desc - mes_formes.min_f
                desc = desc/(mes_formes.max_f-mes_formes.min_f)
                print("? = ", desc)
                frame = np.zeros((600, 600), np.uint8)
                contours[0] = normalise_contour(contours[0], 100)
                contours.append(contours[0])
                cv.drawContours(forme_retenue, contours, 1, 255, -1)
                cv.imshow("Formes retenue", forme_retenue)
                for idx, classifieur in  enumerate(modele_stat):
                    ret, classe = classifieur.predict(desc)
                    if classe.shape[0]*classe.shape[1] != 1:
                        pos = int(ret)
                    else:
                        pos = int(classe[0])
                    print(classifieur.getDefaultName(), " \tclasse = ", classe, " pos= ", pos)
                    frametmp = np.zeros((700, 700), np.uint8)
                    cv.drawContours(frametmp, mes_formes.formes, pos, 255, -1)
                    cv.putText(frametmp, classifieur.getDefaultName(), (40, 40),
                               cv.FONT_HERSHEY_SIMPLEX, 2, 128, 3)
                    rec_dst = (int(idx % 3 * 200), int(idx // 3 * 200), 200, 200)
                    frametmp = cv.resize(frametmp, (200, 200))
                    frame[rec_dst[0]:rec_dst[0]+200, rec_dst[1]:rec_dst[1]+200] = frametmp
                abcdere.efface_feuille = True
                cv.imshow("Classement", frame)
                cv.waitKey(10)
            else:
                frametmp = np.zeros((700, 700), np.uint8)
                cv.putText(frametmp, "Erreur plusieurs contours", (40, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 2, 128, 3)
                cv.imshow("Cela ressemble ", frametmp)
                cv.waitKey(10)
                abcdere.efface_feuille = True
