#!/usr/bin/env python3

from facedb_utils import *
import math
import sys
import time
import cv2
from deepface import DeepFace as DF
from deepface.detectors import DetectorWrapper

import matplotlib.pyplot as plt
from  matplotlib import patches

MODEL = 'Facenet'
DET = 'mtcnn'
if len(sys.argv) > 2:
    MODEL = sys.argv[2]
    if len(sys.argv) > 3:
        DET = sys.argv[3]

model = DF.modeling.build_model(MODEL)
target_size = (model.input_shape[1], model.input_shape[0])

if len(sys.argv) > 1:
    file = sys.argv[1]
    print()
    print(file)
    start  = time.time()

    img = cv2.imread(file)
    
    face_objs = DetectorWrapper.detect_faces(
        detector_backend=DET,
        img=img,
        align=False,
    )

    pltsize = max(2,int(math.sqrt(len(face_objs)))+1)

    fig, axes = plt.subplots(pltsize, pltsize, figsize=(target_size[0]/96*2,target_size[1]/96*2))
    idx = 0
    
    for face in face_objs:
        neyes = 0
        if face.facial_area.left_eye:
            neyes += 1
        if face.facial_area.right_eye:
            neyes += 1
        print(f"{idx} -> {int(round(100*face.confidence))}% - @({face.facial_area.x},{face.facial_area.y} @ {face.facial_area.w}x{face.facial_area.h}) w/{neyes} eye{'s' if neyes != 1 else ''}")
        imgrot = format_image(align_face(face, img)[0], target_size)
        
        x = idx%pltsize
        y = idx//pltsize
        axes[y,x].imshow(imgrot)
        idx += 1
    
    for x in range(pltsize):
        for y in range(pltsize):
            axes[y,x].axis('off')  # Hide axes for cleaner look 

    plt.show()
    