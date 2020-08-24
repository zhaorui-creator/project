import cv2

img=cv2.imread('/home/dl/Downloads/indoorCVPR_09/roiresize/arm.png')
img1=cv2.resize(img,(20,20),interpolation=cv2.INTER_LINEAR)
cv2.imwrite('/home/dl/Downloads/indoorCVPR_09/roiresize1/'+'arm.png',img1)