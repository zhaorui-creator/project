import cv2
import numpy as np
import os
source_path='/home/luo/rui/images/'
#source_path='/home/dl/Downloads/indoorCVPR_09/images/'
#source_path='/home/dl/Downloads/Caltech-UCSD Birds/Caltech-UCSD Birds-200 2010/images/images/001.Black_footed_Albatross/'

target_path='/home/dl/Downloads/indoorCVPR_09/combinesingletest/'
orgimglist=os.listdir(source_path)
print(orgimglist)
orgimglist.sort(key=lambda x: int(x.split('.')[0]))
i=0
for file in orgimglist:
    i=i+1
    img=cv2.imread(source_path+file)
    imgcombine=np.concatenate((img,img),axis=1)
    cv2.imwrite(target_path+str(i)+'.jpg',imgcombine)
