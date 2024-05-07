#!/usr/bin/env python3

import os
import sys
import time
import cv2
import json
import numpy as np
import base64
from deepface.detectors import DetectorWrapper

MODEL = 'mtcnn'
dbdir = '/data/family-photos/facedb'
if dbdir[-1] != '/':
    dbdir += '/'

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

if len(sys.argv) > 2:
    skip = sys.argv[2]
    found = False
    for i in range(len(files)):
        if files[i][0] == skip:
            print('Skipping',i-1,"to get to",skip)
            files = files[i:]
            found = True
            break
    if not found:
        print('Did not find skip-to-file',skip)
        sys.exit(1)

for file,basename in files:
    print()
    print(file)
    start  = time.time()

    n = 0
    out = []

    img = cv2.imread(file)

    face_objs = DetectorWrapper.detect_faces(
        detector_backend=MODEL,
        img=img,
        align=False,
    )

    grayed = False

    for fo in face_objs:
        neyes = 0
        if fo.facial_area.left_eye:
            neyes += 1
        if fo.facial_area.right_eye:
            neyes += 1
        print(f"    {n}: {int(round(100*fo.confidence))}% - @({fo.facial_area.x},{fo.facial_area.y} @ {fo.facial_area.w}x{fo.facial_area.h}) w/{neyes} eye{'s' if neyes != 1 else ''}")

        n += 1
        
        out.append({
            "embedding" : None,
            "facial_area": {
                "x": int(fo.facial_area.x),
                "y": int(fo.facial_area.y),
                "w": int(fo.facial_area.w),
                "h": int(fo.facial_area.h),
                "left_eye": fo.facial_area.left_eye,
                "right_eye": fo.facial_area.right_eye,
            },
            "confidence" : round(fo.confidence, 2),
        })

    jf = basename
    xf = os.path.basename(file)
    if xf != jf:
        jf += '-' + xf
    jf += '.json'
    if out:
        open(dbdir + '/' + jf, 'w').write(json.dumps({
            "path":file,
            "basename":basename,
            "det_algo":MODEL,
            "algo":None,
            "faces":out
        }, indent=2,))
    else:
        if os.path.exists(dbdir + '/' + jf):
            os.unlink(dbdir+'/'+jf)
    
    end = time.time()
    print(f'Processed {n} faces in {end - start}')
