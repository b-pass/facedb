#!/usr/bin/env python3

from deepface import DeepFace as DF
from deepface.modules import preprocessing as DFpp
from deepface.detectors import DetectorWrapper
import os
import sys
import time
from PIL import Image
import cv2
import pickle
import json
import numpy as np
import base64

MODEL = 'Facenet'

model = DF.modeling.build_model(MODEL)
target_size = (model.input_shape[1], model.input_shape[0])

def align_face(face,fullimg):
    (elx,ely) = face.facial_area.left_eye if face.facial_area.left_eye is not None else (0,0)
    (erx,ery) = face.facial_area.right_eye if face.facial_area.right_eye is not None else (0,0)

    # make sure to null-out the source if it should've been in the first place.
    if elx == 0 and ely == 0: face.facial_area.left_eye = None
    if erx == 0 and ery == 0: face.facial_area.right_eye = None

    # no eyes? no align.
    if face.facial_area.left_eye is None or face.facial_area.right_eye is None:
        return face.img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if fullimg.shape[0] == 0 or fullimg.shape[1] == 0:
        return face.img, 0
    
    angle = float(np.degrees(np.arctan2(ery - ely, erx - elx)))

    # right between the eyes
    center = (elx + (erx - elx)/2, ely + (ery - ely)/2)
    print("angle=",angle,"center=",center)

    # crop first to close-ish to what we are working on, to make the subsequent operations faster
    # we need to allow some pixels beyond the target area 
    # which might be rotated into the frame, and might be used in resampling.
    # this gives a ~2x speedup at the cost of some annoying maths
    marginx = face.facial_area.w
    marginy = face.facial_area.h
    print("crop",(face.facial_area.x - marginx, face.facial_area.y - marginy, face.facial_area.x + face.facial_area.w*2, face.facial_area.y + face.facial_area.h*2))
    imgrot = Image.fromarray(fullimg).crop((face.facial_area.x - marginx, face.facial_area.y - marginy, face.facial_area.x + face.facial_area.w*2, face.facial_area.y + face.facial_area.h*2))

    # rotate by the angle about the center (between the eyes) -- which has been moved because of the crop
    imgrot = imgrot.rotate(angle, resample=Image.BICUBIC, center=(center[0] - (face.facial_area.x - marginx), center[1] - (face.facial_area.y - marginy)))

    # re-crop to just the rotated face area, accounting for the first crop
    imgrot = imgrot.crop((marginx,marginy,marginx+face.facial_area.w, marginy+face.facial_area.h))

    # and return an ndarray
    imgrot = np.array(imgrot)
    return imgrot, angle

if len(sys.argv) > 1:
    file = sys.argv[1]
    print()
    print(file)
    start  = time.time()

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    face_objs = DetectorWrapper.detect_faces(
        detector_backend='yolov8',
        img=img,
        align=False,
    )

    for face in face_objs:
        neyes = 0
        if face.facial_area.left_eye:
            neyes += 1
        if face.facial_area.right_eye:
            neyes += 1
        print(f"    {int(round(100*face.confidence))}% - @({face.facial_area.x},{face.facial_area.y} @ {face.facial_area.w}x{face.facial_area.h}) w/{neyes} eye{'s' if neyes != 1 else ''}")
        imgrot = align_face(face, img)[0]
        
        import matplotlib.pyplot as plt
        from  matplotlib import patches
        
        # Create figure and axes
        fig, ax = plt.subplots()

        ax.imshow(imgrot)

        #rect = patches.Rectangle((face.facial_area.x, face.facial_area.y), face.facial_area.w, face.facial_area.h, linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)

        plt.show()
    sys.exit(1)
    