import cv2
import os
orgimg_path='/home/dl/Downloads/indoorCVPR_09/images1/'
objimg_path='/home/dl/Downloads/indoorCVPR_09/background_and_obj/'
target_path1='/home/dl/Downloads/indoorCVPR_09/images/'

target_path2='/home/dl/Downloads/indoorCVPR_09/images2/'


orgimglist=os.listdir(orgimg_path)
objimglist=os.listdir(objimg_path)

orgimglist.sort(key=lambda x: int(x.split('.')[0]))
objimglist.sort(key=lambda x: int(x.split('_')[0]))


for file1 in orgimglist:
    orgimg=cv2.imread(orgimg_path+file1)
    image1 = cv2.resize(orgimg, (500,500),0,0, cv2.INTER_LINEAR)
    cv2.imwrite(target_path1+file1,image1)


for file2 in objimglist:
    objimg=cv2.imread(objimg_path+file2)
    image2 = cv2.resize(objimg, (500,500),0,0, cv2.INTER_LINEAR)
    cv2.imwrite(target_path2+file2,image2)
