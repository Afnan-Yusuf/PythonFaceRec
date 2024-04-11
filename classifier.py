import numpy as np
from PIL import Image
import cv2
import os

def train_classifier(datadir):
    path = [os.path.join(datadir, f) for f in os.listdir(datadir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        img_numpy = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
        faces.append(img_numpy)
        ids.append(id)
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write('classifier.yml')

train_classifier('dataset')
