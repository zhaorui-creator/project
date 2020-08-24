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



import cv2   #导入openCV包



import random
import itertools
from functools import reduce
import operator
import math

#from argumentation import singleadd



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def generate_center():
    random_list=list(itertools.product(range(0,500),range(0,1000)))
    #print(random_list)
    pointnum_list=[4,5,6,7]
    point_num=random.choice(pointnum_list)
    coords = random.sample(random_list,point_num)
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    org_point = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    org_point=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    org_point.reverse()
    center_list=[]
    for i in org_point:
        i=list(i)
        center_list.append(i)
    return center_list


def contrast_brightness_demo(image, c, b):  # C 是对比度，b 是亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)   #改变像素的API
    return dst



def singleadd(img,p):
    light_list=[100,80,60,40,20]
    #for i in range(1,29):
        #if i<10:
    #source_path='/home/dl/Documents/project/classfication/testimg/org/'
    #target_path='/home/dl/Documents/project/classfication/2/'
        #else:
            #source_path='/home/dl/Documents/project/classfication/obj_and_background/'+'abcd'+str(i)+'/'
            #target_path='/home/dl/Documents/project/classfication/obj_and_background+aug/'+'augresult_abcd'+str(i)+'/'
    if round(np.random.uniform(0, 1), 1) <= p:






        print("--------generate------------")
        #image_list=sorted(os.listdir(source_path))
        #print(image_list)
        #k=1
        #for k in range(0,20):
            #for file in image_list:
        lsPointsChoose=generate_center()
        print(lsPointsChoose)
        sample_light=random.choice(light_list)
        #img=Image.open('/home/dl/rui/project/Dataset/0/1.jpg').convert('RGB')
        #img=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)
        #img=cv2.imread('/home/dl/rui/project/Dataset/0/1.jpg')
        mask=np.zeros(img.shape,np.uint8)
        pts=np.array([lsPointsChoose],np.int32)
        print(pts.shape)

        pts=pts.reshape((-1,1,2))

        #画多边形
        mask=cv2.polylines(mask,[pts],True,(0,255,255))

        #填充多边形
        mask2=cv2.fillPoly(mask,[pts],(255,255,255))

        #颜色反转 得到框外白色  框内黑色的掩膜图像
        height,width,depth=mask2.shape
        mask3=np.zeros((height,width,depth),np.uint8)

        for i in range(0,height):
            for j in range(0,width):
                (b,g,r)=mask2[i,j]
                mask3[i,j]=(255-b,255-g,255-r)
        
        #得到roi部分是黑色，区域部分是原图的图像
        source1=cv2.bitwise_and(mask3,img)
        #得到roi部分有图像，框外部分全是黑色的图像
        roi=cv2.bitwise_and(mask2,img)

        #得到经过对比度增强的改变亮度的roi部分
        lightroi=contrast_brightness_demo(roi,1.2,sample_light)
        #将框外经过对比度增强的部分去掉变成全黑
        roiresult=cv2.bitwise_and(mask2,lightroi)

        #得到最终的增强结果
        result=cv2.add(source1,roiresult)
        #cv2.imwrite('1.jpg',result)
                #k=k+1
        return result






class MyDataset(Dataset):
    def __init__(self,root,mode):
        super(MyDataset,self).__init__()
        self.root=root
        self.mode=mode

        
        





        self.name2label={}   #"sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name]=len(self.name2label.keys())
        print(self.name2label)

        #images,label
        self.load_csv('images.csv')
        self.images,self.labels=self.load_csv('images.csv')

        if mode=='train':
            self.images=self.images[:int(0.8*len(self.images))]
            self.labels=self.labels[:int(0.8*len(self.labels))]
        else:
            self.images=self.images[int(0.8*len(self.images)):]
            self.labels=self.labels[int(0.8*len(self.labels)):]
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

    #def denormalize(self,x_hat):
        #normMean = [0.485, 0.456, 0.406]
        #normStd = [0.229, 0.224, 0.225]
        #x=x_hat+normStd+normMean

    def __getitem__(self,idx):
        #idx [0~len(images)]
        #self.images,self.labels
        img,label=self.images[idx],self.labels[idx]
        #normMean = [0.485, 0.456, 0.406]
        #normStd = [0.229, 0.224, 0.225]
        #normMean = [0.4948052, 0.48568845, 0.44682974]
        #normStd = [0.24580306, 0.24236229, 0.2603115]
        #tf=transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                    #transforms.Resize((int(1000*1.25),int(2000*1.25))),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(15),
                    #transforms.CenterCrop((1000,2000)),
                    #transforms.ToTensor(),
                    #transforms.Normalize(normMean,normStd),
                    #lambda y:y.view(6,1000,1000)
                    #normTransform
                    #])


        #if transforms is None:
        normMean = [0.485, 0.456, 0.406]
        normStd = [0.229, 0.224, 0.225]
        if self.mode=='train':
            tf=transforms.Compose([
                #transforms.RandomHorizontalFlip(p=0.5),
                lambda x:Image.open(x).convert('RGB'), #string path =>image data:
                lambda x:cv2.cvtColor(np.asarray(x),cv2.COLOR_BGR2RGB),
                #transforms.Resize((int(1000*0.5),int(2000*0.5))),
                lambda x:singleadd(x,0.5),
                lambda x:Image.fromarray(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                #lambda x:contrast(x,0.6),
                transforms.Resize((int(500*1.25),int(1000*1.25))),
                transforms.CenterCrop((500,1000)),
                transforms.ToTensor(),
                transforms.Normalize(normMean,normStd),
                lambda y:y.view(6,500,500)
                #normTransform
                ])

            #tf=transforms.Compose([
                #lambda x:cv2.imread(x),
                #lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                #lambda x:np.transpose(x,(2,0,1)),
                #transforms.RandomRotation(15),
                #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                #transforms.Resize((int(500*1.25),int(1000*1.25))),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.CenterCrop((500,1000)),
                #transforms.ToTensor(),
                #lambda x:torch.from_tensor(x)
                #transforms.Normalize(normMean,normStd),
                #lambda y:y.view(6,500,500)
            #])
        else:
            tf=transforms.Compose([
                #transforms.RandomHorizontalFlip(p=0.5),
                lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                lambda x:cv2.cvtColor(np.asarray(x),cv2.COLOR_BGR2RGB),
                lambda x:Image.fromarray(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)),
                #transforms.Resize((int(1000*0.5),int(2000*0.5))),
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
            #tf=transforms.Compose([
                #lambda x:cv2.imread(x),
                #lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                #lambda x:np.transpose(x,(2,0,1)),
                #transforms.Resize((int(500*1.25),int(1000*1.25))),
                #transforms.CenterCrop((500,1000)),
                #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                #transforms.ToTensor(),
                #transforms.Normalize(normMean,normStd),
                #lambda y:y.view(6,500,500)

            #])
        img=tf(img)
        label=torch.tensor(label)
        return img,label



#def main():
    #import visdom
    #import time
    #import torchvision
    #viz=visdom.Visdom()
    #db=MyDataset('/home/dl/Documents/project/classfication/Dataset/','train')
    #x,y=next(iter(db))
    #print('sample:',x.shape,y.shape,y)
    #normMean = [0.485, 0.456, 0.406]
    #normStd = [0.229, 0.224, 0.225]
    #tf=transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #lambda x:Image.open(x).convert('RGB'),  #string path =>image data:
                    #transforms.Resize((int(500*1.25),int(1000*1.25))),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.RandomVerticalFlip(p=0.5),
                    #transforms.RandomRotation(15),
                    #transforms.CenterCrop((500,1000)),
                    #transforms.ToTensor(),
                    #transforms.Normalize(normMean,normStd),
                    #lambda y:y.view(6,500,500)
                    #normTransform
                    #])

    #data=torchvision.datasets.ImageFolder(root='/home/dl/Documents/project/classfication/Dataset/',transform=tf)
    #loader=DataLoader(data,batch_size=32,shuffle=True)
    #for x,y in loader:
        #viz.image(x,nrow=8,win='batch',opts=dict(title='batch'))
        #viz.text(str(y.numpy()),win='label',opts=dict(title='batch_y'))
        #time.sleep(10)
    

    #viz.image(x,win='sample_x',opts=dict(title='sample_x'))
#if __name__=='__main__':
    #main()







