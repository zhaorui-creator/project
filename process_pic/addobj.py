import random
import cv2
import os
import numpy as np


def select_file(fileDir): 
    pathDir = sorted(os.listdir(fileDir))    #取图片的原始路径
    filenumber=len(pathDir)
    sample=random.choice(pathDir)
    print (sample)
    return sample
if __name__=='__main__':
    source_path='/home/luo/rui/images/'
    target_path='/home/luo/rui/backgound_and_obj/'
    img_list=sorted(os.listdir(source_path))
    img_list.sort(key=lambda x: int(x.split('.')[0]))
    print(img_list)
    list=[]
    for file in img_list:
        backname=file.split('.')[0]
        m=random.randint(100,400)
        n=random.randint(100,400)
        center=(m,n)
        source_image=cv2.imread(source_path+file)
        width, height, channels = source_image.shape


        sample = select_file('/home/luo/rui/roiresize/')
               
        obj=cv2.imread('/home/luo/rui/roiresize/'+sample)
        

        mask = 255 * np.ones(obj.shape, obj.dtype)
        
        try:
            mixed_clone = cv2.seamlessClone(obj, source_image, mask, center, cv2.MIXED_CLONE)
        except:
            #list.append(0)
            #continue
            center1 = (250,250)
            mixed_clone = cv2.seamlessClone(obj,source_image,mask,center1,cv2.MIXED_CLONE)
       
        filename,extension=os.path.splitext(sample)

        cv2.imwrite(target_path +file, mixed_clone)
    






