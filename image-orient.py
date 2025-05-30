print("Loading...")
import torch
import clip
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

basic_strings = {
    ("upside down",180):"upside down",
    ("sideways",999):"sideways",
    #("rotated",999):"rotated image",
    #("not wrong",0):"a photo without rotation",
    ("people",0):"people",
    ("landscape",0):"landscape",
    #("clutter",0):"clutter",
    ("blank",0):"blank",
}

basic_labels = [x for x in basic_strings.keys()]
basic_features = model.encode_text(clip.tokenize([x for x in basic_strings.values()]).to(device))
basic_features = basic_features / basic_features.norm(dim=-1, keepdim=True)

def check_image(file):
    image = Image.open(file)

    result = []
    upside = None
    nupside = 0
    best = -1
    ibest = 0
    for angle in [0,90,180,270]:
        img = image 
        if angle:
            img = image.rotate(angle, expand=True)

        img_encoding = model.encode_image(preprocess(img).unsqueeze(0).to(device))
        img_encoding = img_encoding / img_encoding.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100 * img_encoding @ basic_features.T).softmax(dim=1)
        values, indices = similarity[0].topk(1)

        confidence = values[0].item()
        desc,a = basic_labels[indices[0].item()]
        if confidence > best:
            best = confidence
            ibest = len(result)
        if desc == 'upside down':
            upside = len(result)
            nupside += 1
        if a != 999:
            a = (a + angle)%360
        result.append( (desc, a, confidence) )
    
    if result[ibest][1] == 999:
        if nupside == 3:
            for i in range(4):
                if result[i][0] != 'upside down':
                    ibest = i
                    break
        else:
            if nupside != 1:
                print(file,"has bad upsides -> ", result)
                return
            ibest = upside
    
    other = (ibest+2)%4

    '''
    if nupside == 0:
        print(file,"has no upside -> ", result)
        return
    elif nupside == 2:
        print(file,"has two upsides -> ", result)
        return
    
    if nupside == 3:
        downside = None
        for i in range(4):
            if result[i][0] != 'upside down':
                downside = i
                break
    else:
        downside = (upside + 2)%4
    '''

    if result[ibest][1] == result[other][1]:
        if result[ibest][1] != 0:
            print(file, "is rotated", result[ibest][1])
            image.rotate(result[ibest][1], expand=True).save(file)
        else:
            print(file, "is rightside up")
    else:
        print(file, "has angle disagreement; a=",result[ibest][1], "; b=", result[other][1], " all --->", result)
    #print(result)

print("Working...")
import sys
import os
for x in sys.argv[1:]:
    print(x)
    #check_all_rotations(x)
    if os.path.isdir(x):
        for f in sorted(os.listdir(x)):
            check_image(os.path.join(x,f))
    else:
        check_image(x)
    
print()
