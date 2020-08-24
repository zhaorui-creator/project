import torch
import os,glob
import random,csv
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np


import torchvision
#import torchvision.models as models
from torchvision.models import resnet18
from torchsummary import summary
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from collections import OrderedDict
from time import *

import torch.nn.functional as F



def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w//2)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img

def evalute(model,loader):
    correct=0
    total=len(loader.dataset)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
            print(logits)
            print(pred)
        #visual
        
        correct+=torch.eq(pred,y).sum().float().item()
    
    return correct/total

def test():
    with torch.no_grad():
        #source_path='/home/luo/rui/classification/test/1111/'
        source_path='/home/luo/rui/combine/'
        #source_path='/home/luo/rui/classification/test/0000/'
        #source_path='/home/luo/rui/classification/test/000/'
        #source_path='/home/luo/rui/classification/test/1111/'
        #source_path='/home/luo/rui/combine/'
        #source_path='/home/luo/rui/classification/test/0000/'
        #source_path='/home/luo/rui/classification/test/000/'
        #source_path='/home/luo/rui/classification/Dataset/0/'
        #source_path='/home/luo/rui/视频帧测试/2/'
        img_list=(os.listdir(source_path))
        img_list.sort(key=lambda x:int(x.split('.')[0]))
        device=torch.device("cuda")
        normMean = [0.485, 0.456, 0.406]
        normStd = [0.229, 0.224, 0.225]
        tf1=transforms.Compose([
                            #transforms.RandomHorizontalFlip(p=0.5),
                            lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                            #transforms.Resize((int(500*1.25),int(1000*1.25))),
                            #lambda x:pad_to_square(x,0),
                            #transforms.Resize((1900,3800)),
                            #transforms.RandomHorizontalFlip(p=0.5),
                            #transforms.RandomVerticalFlip(p=0.5),
                            #transforms.RandomRotation(15),
                            #transforms.CenterCrop((500,1000)),
                            #transforms.Resize((500,1000)),
                            transforms.ToTensor(),
                            lambda x:pad_to_square(x,0),
                            transforms.Normalize(normMean,normStd),
                            lambda y:y.view(6,1280,1280)
                lambda y:y.view(6,500,500)
                            #normTransform
                            ])
        tf2=transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                    transforms.Resize((int(500*1.25),int(1000*1.25))),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(15),
                    transforms.CenterCrop((500,1000)),
                    #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(normMean,normStd),
                    lambda y:y.view(6,500,500)
                    #normTransform
                    ])
        #img=tf(img)
        pred_list=[]
        total=len(img_list)
        a=0
        for file in img_list:
            
            img=tf1(source_path+file)
            img=img.unsqueeze(0).to(device)
            #img.to(device)
            print(img.device)



        
            #model
            model=resnet18(pretrained=False)
            model.eval()
            path_state_dict='/home/luo/rui/classification/checkpoints1/checkpoint_model_epoch_12_loss0.26721033453941345.pth.tar'
            
            state_dict_load=torch.load(path_state_dict)

            new_state_dict=OrderedDict()
            for k,v in state_dict_load.items():
                namekey=k[7:] if k.startswith('module.') else k
                new_state_dict[namekey]=v

            model.load_state_dict(new_state_dict)
            model.to(device)

            print('loaded from ckpt:')
            logits=model(img)
            pred=logits.argmax(dim=1).item()
            #pred_list.append(pred)
            if pred==1:
                a+=1
            
            print(logits)
            print('filename{}_{}'.format(file,pred))
        b=a/total
        #print(b)

if __name__=='__main__':
    begin_time=time()
    test()
    end_time=time()
    run_time=end_time-begin_time
    print(run_time)




