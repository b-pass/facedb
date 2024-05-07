#!/usr/bin/env python3

from facedb_utils import *
import sys
import time
import cv2
import json
import numpy as np
import base64
from deepface import DeepFace as DF
from deepface.detectors import DetectorWrapper

if len(sys.argv) > 1:
    dbdir = sys.argv[1]
    if len(sys.argv) > 2:
        MODEL = sys.argv[2]

if dbdir[-1] != '/':
    dbdir += '/'
model = DF.modeling.build_model(MODEL)
target_size = (model.input_shape[1], model.input_shape[0])

print("Reading from "+dbdir)
files = []
for f in sorted(os.listdir(dbdir)):
    if f.endswith('.json'):
        files.append((dbdir + f,json.loads(open(dbdir + f, 'r').read().encode('ascii'))))
print('... loaded ' + str(len(files)))

for file,data in files:
    print()
    print(file)
    start  = time.time()

    n = 0
    out = []

    img = cv2.imread(data['path'])
    data['algo'] = MODEL

    for face in data['faces']:
        fo = AttrObj(face)
        fimg,angle = align_face(fo, img)
        fimg = format_image(fimg, target_size)
        fimg = normalize_image(fimg, MODEL)
        embedding = model.find_embeddings(fimg)
        face['embedding'] = base64.b64encode(np.asarray(embedding, dtype='float64').tobytes()).decode('ascii')
        n += 1

    open(file, 'w').write(json.dumps(data))
    
    end = time.time()
    print(f'Processed {n} faces in {end - start}')
