import cv2
import os
#for file in sorted(os.listdir('/home/dl/Downloads/indoorCVPR_09/roiresize/')):
image=cv2.imread('/home/luo/rui/roiresize/5c75176296c75.png')



scale_percent=50
width=int(image.shape[1]*scale_percent/100)
height=int(image.shape[0]*scale_percent/100)
dim=(width,height)
resized=cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)
cv2.imwrite('/home/luo/rui/roiresize/5c75176296c75.png',resized)


#resized=cv2.resize(image,(300,300),interpolation=cv2.INTER_LINEAR)
#cv2.imwrite('/home/dl/Downloads/indoorCVPR_09/roiresize/'+'robot.png',resized)


img_list=os.listdir('/home/luo/rui/classification/Dataset/2/')
print(len(img_list))