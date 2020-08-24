import os

path1='/home/dl/Downloads/indoorCVPR_09/test1/'
path2='/home/dl/Downloads/indoorCVPR_09/test2/'


img_list1=sorted(os.listdir(path1))
img_list2=sorted(os.listdir(path2))
print(img_list1)
print(img_list2)