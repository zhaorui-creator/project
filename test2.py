import torch
import os,glob
import random,csv
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np


import visdom
import time
import torchvision



def main():
    viz=visdom.Visdom()
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    tf=transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                    transforms.Resize((int(500*1.25),int(1000*1.25))),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.CenterCrop((500,1000)),
                    transforms.ToTensor(),
                    transforms.Normalize(normMean,normStd),
                    #lambda y:y.view(6,500,500)
                    #normTransform
                    ])

    data=torchvision.datasets.ImageFolder(root='/home/dl/Documents/project/classfication/Dataset/',transform=tf)
    loader=DataLoader(data,batch_size=32,shuffle=True)
    print(data.class_to_idx)
    for x,y in loader:
        viz.image(x,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch_y'))
        time.sleep(10)


if __name__=='__main__':
    main()