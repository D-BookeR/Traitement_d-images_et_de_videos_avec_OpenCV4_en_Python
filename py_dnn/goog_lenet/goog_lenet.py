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

def choisir_caffe_modele():
    if mon_appli:
        nom_modele = wx.FileSelector("Fichier des poids",
                                     wildcard="poids du modèle (*.caffemodel)|*.caffemodel")
        nom_proto = wx.FileSelector("Fichier de configuration",
                                    wildcard="configuration du modèle (*.prototxt)|*.prototxt")
        nom_classe = wx.FileSelector("Fichier des classes",
                                     default_filename="classification_classes_ILSVRC2012.txt",
                                     wildcard="classes du modèle (*.txt)|*.txt")
    else:
        print("ATTENTION LECTURE DES FICHIERS EN UTILISANT")
        print("LES CHEMINS PAR DEFAUT")
        path = "f:/testDNN/objectdetection/caffe/"
        nom_modele = path + "bvlc_googlenet.caffemodel"
        nom_proto = path + "bvlc_googlenet.prototxt"
        nom_classe = path + "classification_classes_ILSVRC2012.txt"
    return nom_modele, nom_proto, nom_classe

if __name__ == '__main__':
    nom_modele, nom_proto, nom_classe = choisir_caffe_modele()
    caffe = cv.dnn.readNet(nom_modele, nom_proto)
    if caffe.empty():
        print("Le réseau est vide!")
        exit()
    try:
        with open(nom_classe, 'rt') as f:
            classes = f.read().split('\n')
    except:
        classes = None

    video = cv.VideoCapture(CAMAPI)

    code = 0
    while code != CODE_TOUCHE_FIN:
        if video.isOpened():
            ret, img = video.read()
        else:
            if mon_appli is None:
                nom_image = 'mon_image.jpg'
            else:
                nom_image = wx.FileSelector(
                    "Image",
                    wildcard="image jpeg  (*.jpg)|*.jpg|image tiff  (*.tif)|*.tif")
            img_original = cv.imread(nom_image)
            img = img_original.copy()
        if img is not None:
            blob = cv.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=False)
            caffe.setInput(blob)
            val_sorties = caffe.forward()
            for val in val_sorties:
                tri_idx = np.argsort(val)
                indice_classe = tri_idx[-1]
                proba = val[indice_classe]
                cv.putText(img, str(int(proba*100)/100),
                            (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 128), 2)
                if classes is not None:
                    cv.putText(img, classes[indice_classe],
                               (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 128), 2)
                else:
                    cv.putText(img, str(indice_classe),
                               (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 128), 2)
                for idx in tri_idx[-1:-4:-1].tolist():
                    if classes is not None:
                        proba = val[idx]
                        print(str(proba), classes[idx])
                    else:
                        proba = val[idx]
                        print(str(proba), str(idx))

            cv.imshow("Image", img)
            if video.isOpened():
               code = cv.waitKey(10)
            else:
               code = cv.waitKey()
    caffe.dumpToFile("bvlc_googlenet.dot")
    cv.destroyAllWindows()

