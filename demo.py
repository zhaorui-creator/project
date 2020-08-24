import os
import cv2
img_list=os.listdir('/home/luo/rui/background1/')
img_list.sort(key=lambda x:int(x.split('.')[0]))
print(len(img_list))
for i in range(150,200):
    print(i)


#for i in range()