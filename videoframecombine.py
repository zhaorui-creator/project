import cv2
import os
import numpy as np
background_path='/home/luo/rui/background1/'
target_path='/home/luo/rui/combine/'
img_list=os.listdir(background_path)
img_list.sort(key=lambda x:int(x.split('.')[0]))

k=90
j=0
for i in range(149,5348):
    if i<900:
        preFrame=img_list[i-k]


    else:
        preFrame=img_list[i-900]
    
    curFrame=img_list[i]
    print(curFrame)


    
    img1=cv2.imread(background_path+curFrame)
    img2=cv2.imread(background_path+preFrame)


    imgcombine=np.concatenate((img2,img1),axis=1)
    cv2.imwrite(target_path+str(j+1)+'.jpg',imgcombine)
    k=k+1
    j=j+1
