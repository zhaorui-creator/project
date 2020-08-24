import cv2
import numpy as np
import os

target_path='/home/luo/rui/classification/Dataset/1/'

orgimg_path='/home/luo/rui/images/'
objimg_path='/home/luo/rui/backgound_and_obj1/'


orgimglist=os.listdir(orgimg_path)
objimglist=os.listdir(objimg_path)

orgimglist.sort(key=lambda x: int(x.split('.')[0]))
objimglist.sort(key=lambda x: int(x.split('.')[0]))


for i in range(0,15451):
    img1=cv2.imread(orgimg_path+orgimglist[i])
    img2=cv2.imread(objimg_path+objimglist[i])
    img3=np.concatenate((img2,img1),axis=1)
    cv2.imwrite(target_path+str(i+1)+'.jpg',img3)
    print(i)