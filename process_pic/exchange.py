import cv2
import os
import numpy as np

img_path = '/home/luo/rui/classification/Dataset/1/'

def exchange(img_path):

    img_list = os.listdir(img_path)

    img_list.sort(key = lambda x:int(x.split('.')[0]))
    print(img_list)

    newimg = np.zeros((500,1000,3))

    for file in img_list:
        img = cv2.imread(img_path+file)
        newimg[:,:500,:] = img[:,500:1000,:]
        newimg[:,500:1000,:]=img[:,:500,:]
        cv2.imwrite('/home/luo/rui/classification/Dataset/2/'+file,newimg)
        print(file)
        newimg[:,:,:] = 0


if __name__ == '__main__':
    exchange(img_path)