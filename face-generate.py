#!/usr/bin/env python3

from deepface import DeepFace as DF
from deepface.modules import preprocessing as DFpp
import os
import sys
import time
from PIL import Image
import cv2
import pickle
import json
import numpy as np
import base64

dbdir = '/data/family-photos/facedb/'

MODEL = 'Facenet'

#def writeFace(detected_face, filename):    
#    detected_face = detected_face * 255
#    cv2.imwrite(filename, detected_face[:, :, ::-1])

files = []
for f in sorted(os.listdir(sys.argv[1])):
    file = os.path.join(sys.argv[1], f)
    if os.path.isfile(file):
        files.append((file, f))
    elif os.path.isdir(file):
        ext = ''
        if (x := file.rfind('.')) > 0:
            ext = file[x+1:]
        if ext.lower() != 'mp4':
            print(f'Ignored dir {f} ({ext})')
            continue
        mp4 = f
        mp4dir = file
        for f in sorted(os.listdir(mp4dir)):
            file = mp4dir + '/' + f
            if os.path.isfile(file):
                files.append((file, mp4))

model = DF.modeling.build_model(MODEL)
target_size = (model.input_shape[1], model.input_shape[0])
for file,basename in files:
    print()
    print(file)
    start  = time.time()
    
    faces = DF.detection.extract_faces(file, detector_backend='yolov8', enforce_detection=False, target_size=target_size)
    n = 0
    out = []
    for face in faces:
        if face['confidence'] < 0.25:
            continue

        print(f"    {n}: {int(round(100*face['confidence']))}% - {face['facial_area']}")
        n += 1
        
        img = DFpp.normalize_input(img=face["face"], normalization='Facenet')
        emb1 = model.find_embeddings(img)
        emb2 = model.find_embeddings(np.fliplr(img))
        embedding = np.append(emb1, emb2)
        out.append({
            "embedding" : base64.b64encode(np.asarray(embedding, dtype='float64').tobytes()).decode('ascii'),
            "facial_area" : face['facial_area'],
            "confidence" : face['confidence'],
        })

    if out:
        jf = basename
        xf = os.path.basename(file)
        if xf != jf:
            jf += '-' + xf
        jf += '.json'
        open(dbdir + '/' + jf, 'w').write(json.dumps({
            "path":file,
            "basename":basename,
            "algo":MODEL,
            "faces":out
        }, indent=2,))
    end = time.time()
    print(f'Processed {n} faces in {end - start}')

#DF.find()
