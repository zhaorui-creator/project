import cv2
import numpy as np
import os

#orgimage_path='/home/dl/result/org/'
#augimage_path='/home/dl/result/augorg/'

#target_path='/home/dl/Data/org_with_augorg/'
#orgimage_list=sorted(os.listdir(orgimage_path))
#augimage_list=sorted(os.listdir(augimage_path))
#print(orgimage_list)
#print(augimage_list)

#for i in range(0,28):
    #img1=cv2.imread(orgimage_path+orgimage_list[i])
    #img2=cv2.imread(augimage_path+augimage_list[i])

    #img3=np.concatenate((img1,img2),axis=1)
    #if i<9:
        #cv2.imwrite(target_path+'0_nonenoise_passage'+str(0)+str(i+1)+'_with_existnoise.jpg',img3)
    #else:
        #cv2.imwrite(target_path+'0_nonenoise_passage'+str(i+1)+'with_existnoise.jpg',img3)
orgimage='/home/dl/Documents/project/classfication/testimg/org/orgimg.jpg'
target_path='/home/dl/Documents/project/classfication/org_and_orgnoise/'
#for m in range(0,28):
    #if m<9:
objimage_path='/home/dl/Documents/project/classfication/2/'
        #orgimage='/home/dl/result/org/augabcd'+str(0)+str(m+1)+'.jpg'
    #else:
        #objimage_path='/home/dl/result/obj_and_background+aug/augresult_abcd'+str(m+1)+'/'
        #orgimage='/home/dl/result/org/augabcd'+str(m+1)+'.jpg'

objimage01_list=sorted(os.listdir(objimage_path))

print(objimage01_list)


i=1
for objfile in objimage01_list:
            
    #objname=objfile.split('.')
    #objname=objname[0].split('_')[-1][0:-2]
    #print(objname)
    img1=cv2.imread(orgimage)
    img2=cv2.imread(objimage_path+objfile)

    img3=np.concatenate((img1,img2),axis=1)
    #if m<9:
    cv2.imwrite(target_path+str(i)+'.jpg',img3)
    #else:
        #cv2.imwrite(target_path+'1_existnoisepassage'+str(m+1)+'_with_existnoise'+objname+str(i)+'.jpg',img3)
    
    i=i+1
    