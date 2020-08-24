import numpy as np
import cv2
from skimage import data, filters


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



device=torch.device("cuda")
normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]

#cap = cv2.VideoCapture('/home/dl/data/subway/hongqiao.mp4')

#videoinput1 = cv2.VideoCapture("/home/dl/data/滞留物/hiv00009.mp4")
videoinput1 = cv2.VideoCapture('/home/luo/rui/0413_01.mp4')

counter = 0

initial_frames = []

initialized_flag = False

frames = 0
while(True):
    
    ret1, img1 = videoinput1.read()#(960, 1280, 3) #720

    #black_img = np.zeros((960,1280,3))
    #density_img = np.zeros((723,500,3))
    if img1 is None:
        break
    
    #img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    
    if len(initial_frames) < 400:
        if counter % 10 == 0:
            initial_frames.append(img1)
            frames += 1
    elif initialized_flag == False: # 确保初始化只执行一次
        #print("run to here 1")
        #frames = np.concatenate(initial_frames)
        sequence = np.stack(initial_frames, axis=3)
        
        print(sequence.shape)
        medianFrame = np.median(sequence, axis=3)#.astype(dtype=np.uint8) 
        
        background = medianFrame.astype(dtype=np.uint8)
        
        initialized_flag = True # 确保初始化只做一次
        
    #print(len(initial_frames))
    #如果还没有初始化，下面的不用处理
    #print(initialized_flag)
    if initialized_flag == True:
        #更新背景
        #print("run to here 2")
        if counter % 20 == 0:
            
           
            medianFrame = 0.98*medianFrame + 0.02*img1
            
            #background = medianFrame.astype(dtype=np.uint8) 
            
            #initial_frames.pop(0)
            #initial_frames.append(img1)
            #sequence = np.stack(initial_frames, axis=2)
            #medianFrame = np.median(sequence, axis=2)#.astype(dtype=np.uint8) 
            background = medianFrame.astype(dtype=np.uint8)
            #cv2.imwrite('/home/luo/rui/frames/'+str(counter)+'.jpg',background)
            
        background_show =  cv2.resize(background, (int(background.shape[1] / 2), int(background.shape[0] / 2)))
        img1 =  cv2.resize(img1, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)))
        
        #print(background_show.shape,img1.shape)
        
        imgshow = np.concatenate([background_show,img1],axis=1)
        imgshow = cv2.cvtColor(imgshow, cv2.COLOR_BGR2RGB)


        
        tf1=transforms.Compose([
                            #transforms.RandomHorizontalFlip(p=0.5),
                            #lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                            #transforms.Resize((int(500*1.25),int(1000*1.25))),
                            transforms.Resize((500,1000)),
                            #transforms.RandomHorizontalFlip(p=0.5),
                            #transforms.RandomVerticalFlip(p=0.5),
                            #transforms.RandomRotation(15),
                            #transforms.CenterCrop((500,1000)),
                            #transforms.Resize((500,1000)),
                            transforms.ToTensor(),
                            transforms.Normalize(normMean,normStd),
                            lambda y:y.view(6,500,500)
                            #normTransform
                            ])



        img=tf1(imgshow)
        img=img.unsqueeze(0).to(device)
        #img.to(device)
        print(img.device)



    
        #model
        model=resnet18(pretrained=False)
        path_state_dict='./best8.mdl'
        
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
        
        #cv2.imshow("11",imgshow)
        #cv2.waitKey(10)
        
    #else:
    #    print("run to here 3")
    
    counter += 1