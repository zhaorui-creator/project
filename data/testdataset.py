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
from torchvision.models import resnet50
from torchsummary import summary
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader






class testdataset(Dataset):
    def __init__(self,root,mode):
        super(testdataset,self).__init__()
        self.root=root
        #self.transform=transform

        self.name2label={}   #"sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name]=len(self.name2label.keys())
        print(self.name2label)

        #images,label
        self.load_csv('images.csv1')
        self.images,self.labels=self.load_csv('images1.csv')

        #if mode=='train':
            #self.images=self.images[:int(0.8*len(self.images))]
            #self.labels=self.labels[:int(0.8*len(self.labels))]
        #else:
            #self.images=self.images[int(0.8*len(self.images)):]
            #self.labels=self.labels[int(0.8*len(self.labels)):]
        #else:
            #self.images=self.images[int(0.8*len(self.images)):]
            #self.labels=self.labels[int(0.8*len(self.labels)):]




    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images=[]
            for name in self.name2label.keys():
                images+=glob.glob(os.path.join(self.root,name,'*jpg'))
            print(len(images),images)

            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer=csv.writer(f)
                for img in images:#'/home/dl/dataset/1/1_nonenoisepassage22_with_existnoisefridge17.jpg'
                    name=img.split('/')[-2]
                    label=self.name2label[name]
                    #'/home/dl/dataset/1/1_nonenoisepassage22_with_existnoisefridge17.jpg' 1
                    writer.writerow([img,label])
                print('writen into csv file:',filename)

        #read from csv file
        images,labels=[],[]
        with open(os.path.join(self.root,filename)) as f:
            reader=csv.reader(f)
            for row in reader:
                img,label=row
                label=int(label)
                images.append(img)
                labels.append(label)
        
        assert len(images)==len(labels)
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img,label=self.images[idx],self.labels[idx]
        normMean = [0.485, 0.456, 0.406]
        normStd = [0.229, 0.224, 0.225]
        tf=transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                    #transforms.Resize((int(500*1.25),int(1000*1.25))),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(15),
                    #transforms.CenterCrop((500,1000)),
                    transforms.Resize((500,1000)),
                    transforms.ToTensor(),
                    transforms.Normalize(normMean,normStd),
                    lambda y:y.view(6,500,500)
                    #normTransform
                    ])
        #img=tf(img)
        img=tf(img)
        label=torch.tensor(label)
        return img,label


def main():
    db=testdataset('/home/dl/Documents/project/classfication/test/','test')





if __name__=='__main__':
    main()
