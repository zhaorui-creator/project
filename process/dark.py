import cv2
import numpy as np
import os,random
def darkpic(file_path,light_ratio):
    img=cv2.imread(file_path)
    w=img.shape[1]
    h=img.shape[0]

    #all dark
    for x in range(0,w):
        for y in range(0,h):
            #将像素值整体减少，设为原来像素的20%
            img[y,x,0]=int(img[y,x,0]*light_ratio)
            img[y,x,1]=int(img[y,x,1]*light_ratio)
            img[y,x,2]=int(img[y,x,2]*light_ratio)

        #显示进度条
        if x%10==0:
            print('.')
    return img
    



if __name__=='__main__':
    light_ratio_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
        #light_ratio=random.choice(light_ratio_list)
        #print(light_ratio)
    source_path='/home/dl/Dataset/org1/'
    target_path='/home/dl/Documents/project/classfication/1/'
    sourceimg_list=sorted(os.listdir(source_path))
    i=0
    for file in sourceimg_list:
        i=i+1
        for j in range(0,10):
            j=j+1
            light_ratio=random.choice(light_ratio_list)
            print(light_ratio)
            img = darkpic(source_path+file,light_ratio)
            if i<10:
                cv2.imwrite(target_path+str(i)+'/'+'abcd'+str(i)+str(i)+'dark'+str(j)+'.jpg',img)
            else:
                cv2.imwrite(target_path+str(i)+'/'+'abcd'+str(i)+'dark'+str(j)+'.jpg',img)


    #darkpic('/home/dl/Documents/project/classfication/resize/abcd01.jpg','/home/dl/Documents/project/classfication/1/')
