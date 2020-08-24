from PIL import Image
from torchvision import transforms
import numpy as np
import torch

import torch.nn.functional as F

def horisontal_flip(images,targets):
    images = torch.flip(images,[-1])
    targets[:,2]=1-targets[:,2]
    return images,targets
target_path2='/home/dl/Downloads/PyTorch-YOLOv3/data/coco/images/train2014/COCO_train2014_000000000009.jpg'
img=transforms.ToTensor()(Image.open(target_path2).convert('RGB'))
boxes=torch.from_numpy(np.loadtxt('/home/dl/Downloads/PyTorch-YOLOv3/data/coco/labels/train2014/COCO_train2014_000000000009.txt'))
targets=torch.zeros((len(boxes),6))
targets[:,1:]=boxes
print(targets)

#img,targets=horisontal_flip(img,targets)

targets = torch.cat(targets, 0)
#print(img)
print(targets)