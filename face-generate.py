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

#from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

dbdir = '/data/family-photos/facedb/'

MODEL = 'Facenet'

model = DF.modeling.build_model(MODEL)
target_size = (model.input_shape[1], model.input_shape[0])

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
    
    imgrot = Image.fromarray(fullimg)
    
    angle = float(np.degrees(np.arctan2(ery - ely, erx - elx)))

    # right between the eyes
    center = (elx + (erx - elx)/2, ely + (ery - ely)/2)

    # crop first to close-ish to what we are working on, to make the subsequent operations faster
    # we need to allow some pixels beyond the target area which might be rotated into the frame, and might be used in resampling.
    # this gives a ~2x speedup at the cost of some annoying maths, but PIL allows these numbers to be negative and go outside the original image, cool.
    marginx = face.facial_area.w
    marginy = face.facial_area.h
    imgrot = imgrot.crop((face.facial_area.x - marginx, face.facial_area.y - marginy, face.facial_area.x + face.facial_area.w + marginx, face.facial_area.y + face.facial_area.h + marginy))

    # rotate by the angle about the center (between the eyes) -- which has been moved because of the crop
    imgrot = imgrot.rotate(angle, resample=Image.BICUBIC, center=(center[0] - (face.facial_area.x - marginx), center[1] - (face.facial_area.y - marginy)))

    # re-crop to just the rotated face area, accounting for the first crop
    imgrot = imgrot.crop((marginx, marginy, marginx+face.facial_area.w, marginy+face.facial_area.h))

    # and return an ndarray
    imgrot = np.array(imgrot)
    return imgrot, angle

def format_image(current_img):
    # taken from DeepFace.detection.extract_faces
    factor_0 = target_size[0] / current_img.shape[0]
    factor_1 = target_size[1] / current_img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(current_img.shape[1] * factor),
        int(current_img.shape[0] * factor),
    )
    current_img = cv2.resize(current_img, dsize, interpolation=cv2.INTER_CUBIC)

    diff_0 = target_size[0] - current_img.shape[0]
    diff_1 = target_size[1] - current_img.shape[1]
    
    # Put the base image in the middle of the padded image
    current_img = np.pad(
        current_img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )
    
    # double check: if target image is not still the same size with target.
    if current_img.shape[0:2] != target_size:
        current_img = cv2.resize(current_img, target_size, interpolation=cv2.INTER_CUBIC)

    # normalizing the image pixels
    # what this line doing? must?
    img_pixels = image.img_to_array(current_img)
    img_pixels = np.expand_dims(img_pixels, axis=0)

    # Facenet normalization:
    img_pixels = (img_pixels - img_pixels.mean()) / img_pixels.std()

    # Faceenet2018 normalization:
    # img /= 127.5
    # img -= 1

    return img_pixels

for file,basename in files:
    print()
    print(file)
    start  = time.time()

    n = 0
    out = []

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    face_objs = DetectorWrapper.detect_faces(
        detector_backend='yolov8',
        img=img,
        align=False,
    )

    for fo in face_objs:
        neyes = 0
        if fo.facial_area.left_eye:
            neyes += 1
        if fo.facial_area.right_eye:
            neyes += 1
        print(f"    {n}: {int(round(100*fo.confidence))}% - @({fo.facial_area.x},{fo.facial_area.y} @ {fo.facial_area.w}x{fo.facial_area.h}) w/{neyes} eye{'s' if neyes != 1 else ''}")

        if not fo.facial_area.left_eye or not fo.facial_area.right_eye:
            if fo.confidence < 0.75:
                print("     ... skipped (no eyes, low confidence)")
                continue
        else:
            if fo.confidence < 0.5:
                print("     ... skipped (low confidence)")
                continue

        fimg,angle = align_face(fo,img)
        fimg = format_image(fimg)
    
        n += 1
        
        embedding = model.find_embeddings(fimg)
        #emb2 = model.find_embeddings(np.fliplr(img))
        #embedding = np.append(embedding, emb2)
        out.append({
            "embedding" : base64.b64encode(np.asarray(embedding, dtype='float64').tobytes()).decode('ascii'),
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
            "algo":MODEL,
            "faces":out
        }, indent=2,))
    else:
        if os.path.exists(dbdir + '/' + jf):
            os.unlink(dbdir+'/'+jf)
    
    end = time.time()
    print(f'Processed {n} faces in {end - start}')
