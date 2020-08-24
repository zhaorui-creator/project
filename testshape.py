from PIL import Image
import numpy as np
import cv2
import torch

from torchvision import transforms
normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
img = '/home/luo/rui/classification/test/testshape/17.jpg'
tf1=transforms.Compose([
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #lambda x:Image.open(x).convert('RGB'), #string path =>image data:
                        lambda x:cv2.imread(x),
                        lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                        #lambda x:np.transpose(x,(2,0,1)),
                        #transforms.Resize((int(500*1.25),int(1000*1.25))),
                        #lambda x:pad_to_square(x,0),
                        #transforms.Resize((1900,3800)),
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomVerticalFlip(p=0.5),
                        #transforms.RandomRotation(15),
                        #transforms.CenterCrop((500,1000)),
                        #transforms.Resize((500,1000)),
                        transforms.ToTensor(),
                        #lambda x:torch.from_numpy(x).float(),
                        transforms.Normalize(normMean,normStd),
                        #lambda y:y.view(6,1920,1920)
                        #normTransform
                        ])



tf=transforms.Compose([
                #transforms.RandomHorizontalFlip(p=0.5),
                lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                #transforms.Resize((int(1000*0.5),int(2000*0.5))),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomRotation(15),
                #transforms.CenterCrop((500,1000)),
                #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                #transforms.Normalize(normMean,normStd),
                #lambda y:y.view(6,500,500)
                #normTransform
                ])
img=tf(img)
print(img.shape)
