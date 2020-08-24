import cv2
import os

source_path='/home/luo/rui/roi/'
target_path='/home/luo/rui/roiresize/'
img_list=sorted(os.listdir(source_path))

for file in img_list:
    image=cv2.imread(source_path+file)
    scale_percent=20
    width=int(image.shape[1]*scale_percent/100)
    height=int(image.shape[0]*scale_percent/100)
    dim=(width,height)
    resized=cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(target_path+file,resized)


