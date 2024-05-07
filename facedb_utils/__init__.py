
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
tensorflow.keras.utils.disable_interactive_logging()

from deepface.modules import preprocessing as DFpp
import sys
from PIL import Image
import cv2
import numpy as np

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

def format_image(current_img, target_size):
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
    
    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)

    return current_img

def normalize_image(img, model):
    # about to be converted, so do that
    img = img.astype('float64')

    # do model-specific normalization
    if model == "Facenet":
        for c in range(3):
            x = img[..., c]
            img[..., c] = (x - x.mean()) / x.std()

    elif model == "Facenet2018":
        # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
        img /= 127.5
        img -= 1

    elif model == "VGGFace":
        # The training images were rescaled such that the smaller of width and height was equal
        # to 256. During training, the network is fed with random 224 × 224 pixel patches cropped
        # from these images (where crops change every time an image is sampled). The data was fur-
        # ther augmented by flipping the image left to right with 50% probability; however, we did not
        # perform any colour channel augmentation.
        pass

    elif model == "VGGFace2":
        # Training implementation details. During training, a region
        # of 224 × 224 pixels is randomly cropped from each sample,
        # and the shorter side is resized to 256. The mean value of
        # each channel is subtracted for each pixel. Transformation
        # to monochrome augmentation is used with a probability
        # of 20% in order to reduce over-fitting on colour images.
        for c in range(3):
            img[..., c] -= img[..., c].mean()

    elif model == "ArcFace":
        # Same as CosFace: The faces are cropped and resized to 112×112,
        # and each pixel (ranged between [0, 255]) in RGB images is normalised
        # by subtracting 127.5 then divided by 128.
        img -= 127.5
        img /= 128
    
    else:
        img = img / img.max()
    
    # change from [h, w, 3] to [1, h, w, 3]
    img = np.expand_dims(img, axis=0)
    return img

class AttrObj:
    def __init__(self, d):
        for (k,v) in d.items():
            if type(v) is dict:
                setattr(self, k, AttrObj(v))
            else:
                setattr(self, k, v)
