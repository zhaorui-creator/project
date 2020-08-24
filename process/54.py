import cv2
import numpy as np
from math import sqrt
import os
import random
import shutil
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
center_list7=[]
list1=[]

boundcenter=[[(60,224),(173,24),(420,497),(3,499),(5,299)],
            [(296,151),(307,149),(494,491),(8,490),(4,337)],
            [(113,293),(153,293),(279,490),(41,498)],
            [(255,235),(282,235),(373,490),(238,490)],
            [(226,296),(258,294),(373,492),(115,493)],
            [(282,262),(495,301),(496,449),(285,327)],
            [(165,375),(183,376),(237,460),(137,461)],
            [(42,190),(67,190),(327,495),(9,497),(3,365)],
            [(3,272),(22,272),(189,466),(91,495),(7,495)],
            [(279,233),(327,232),(265,470),(75,472)],
            [(190,299),(221,300),(454,494),(41,494)],
            [(446,319),(470,320),(496,400),(495,494),(326,494)],
            [(200,222),(230,223),(414,494),(155,495)],
            [(86,236),(96,236),(243,488),(57,491)],
            [(301,265),(312,265),(244,452),(162,427)],
            [(395,326),(416,325),(237,494),(249,494)],
            [(214,269),(239,266),(311,49),(180,492)],
            [(356,282),(377,284),(434,492),(214,495)],
            [(379,247),(413,247),(498,499),(270,491)],
            [(244,263),(273,267),(391,496),(160,496)],
            [(271,306),(496,311),(495,496),(138,497)],
            [(255,295),(273,297),(428,493),(121,494),(206,344)],
            [(232,346),(280,348),(323,488),(181,488)],
            [(255,295),(273,297),(428,493),(121,494),(206,344)],
            [(232,346),(280,348),(325,488),(181,488)],
            [(235,280),(264,279),(493,426),(495,495),(4,493),(3,412)],
            [(4,309),(346,208),(365,493),(6,492)],
            [(178,305),(306,304),(491,357),(495,496),(5,496),(4,350)],
            [(144,197),(341,193),(491,243),(493,496),(7,496),(4,254)],
            [(86,382),(304,382),(339,457),(408,464),(414,496),(10,495)],
            [(117,342),(149,344),(205,497),(81,497)],
            [(45,426),(175,423),(208,497),(7,496)],
            [(106,320),(160,322),(160,496),(6,495),(3,434),(37,377),(70,377),(106,322)],
            [(146,346),(493,323),(494,496),(131,496)],
            [(339,307),(485,304),(498,342),(495,493),(295,494)],
            [(286,354),(384,348),(404,497),(123,497)],
            [(314,233),(435,229),(498,382),(497,493),(139,492)],
            [(233,376),(285,374),(323,496),(157,495)],
            [(361,384),(430,385),(441,493),(152,492),(253,430),(316,434)],
            [(163,365),(411,359),(452,495),(34,496)],
            [(371,227),(428,229),(471,495),(146,496)],
            [(156,381),(243,380),(263,392),(366,402),(496,496),(106,497),(157,380)],
            [(185,403),(329,402),(404,495),(110,495)],
            [(202,408),(300,407),(397,493),(94,493)],
            [(138,283),(179,281),(490,393),(494,496),(58,493)],
            [(251,293),(265,293),(399,485),(95,483)],
            [(269,243),(305,248),(494,496),(4,497),(4,361)],
            [(232,389),(282,389),(338,492),(184,492)],
            [(56,262),(93,254),(302,496),(80,499)],
            [(135,181),(188,182),(319,495),(44,496),(161,371),(140,359)],
            [(241,368),(273,370),(333,492),(181,493)],
            [(380,319),(390,319),(492,493),(351,491)],
            [(77,393),(191,390),(217,493),(78,496)],
            [(236,437),(284,435),(420,458),(441,498),(107,498),(109,465)],
            [(411,229),(455,233),(495,404),(496,497),(332,495)],
            [(185,385),(296,389),(336,496),(143,495)],
            [(351,264),(379,264),(466,496),(343,496)],
            [(241,298),(319,301),(320,342),(354,343),(412,494),(299,494),(286,357),(239,349)],
            [(460,297),(495,306),(476,496),(272,459),(380,391),(388,377)],
            [(242,327),(277,330),(240,498),(7,494),(5,405),(129,430)],
            [(34,308),(96,306),(178,491),(17,494)]

            



]


polygon=Polygon(boundcenter[53])
def select_file(fileDir): 
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber=len(pathDir)
    sample=random.choice(pathDir)
    print (sample)
    return sample


for m in range(0,430,25):
    for n in range(0,450,25):
        center_list7.append((m,n))
for i in range(0,len(center_list7)):
    print(len(center_list7))
    center=center_list7[i]
    center1=Point(center_list7[i][0],center_list7[i][1])
    print(center1)
    if center1.within(polygon):
        list1.append(center_list7[i])
print(len(list1))

if __name__=='__main__':

    source_path='/home/dl/Dataset/org/54/'
    #obj_path='/home/dl/resize/'
    target_path='/home/dl/Dataset/background_and_obj/54/'
    source_image=cv2.imread(source_path+'abcd54.jpg')
    #obj=cv2.imread(obj_path+'roadblock.png')

    #mask = 255 * np.ones(obj.shape, obj.dtype)
 
    # The location of the center of the src in the dst
    width, height, channels = source_image.shape
    center_list1=[(62,62),(187,62),(312,62),(437,62),(62,187),(187,187),(312,187),(437,187),(62,312),
                (187,312),(312,312),(437,312),(62,437),(187,437),(312,437),(437,437)]
    center_list2=[(125,125),(375,125),(170,375),(375,375)]
    center_list3=[(50,200),(55,250),(60,300),(65,350)]
    center_list4=[(40,360)]
    center_list5=[(125,125),(375,125),(125,375),(375,375)]
    center_list6=[(40,250)]
    center_list7=[]
    #for m in reversed(range(100,370,25)):
        #for n in range(100,450,15):
            #center_list7.append((m,n))

    #for m in range(0,430,25):
        #for n in range(0,450,25):
            #center_list7.append((m,n))

    #m=40
    #for n in range(250,400,10):
        #center_list7.append((m,n))

    #for m in range(40,451):
        #for n in range(40,451):
            #center_list6.append((m,n))

    #for i in range(0,len(center_list7)):

        #center = center_list7[i]
    #for i in range(0,1):
        #center=center_list6[i]
    list=[]
    for j in range(0,len(list1)):
            print(len(list1))
            center=list1[j]
#center = (64，)
#center = (85,192)
#center = (85,256)
#center = (height // 2-90, width // 2-40)
#center = (height // 2-90, width // 2-40)
#center = (height // 2-90, width // 2-40)
#center = (height // 2-90, width // 2-40)
#center = (height // 2-90, width // 2-40)


 
# Seamlessly clone src into dst and put the results in output
            sample = select_file('/home/dl/roi1/')
            list.append(sample)
        #obj=cv2.imread('/home/dl/roi1/'+sample)
            obj=cv2.imread('/home/dl/roi1/'+sample)

            mask = 255 * np.ones(obj.shape, obj.dtype)
        #normal_clone = cv2.seamlessClone(obj, source_image, mask, center, cv2.NORMAL_CLONE)
            try:
                mixed_clone = cv2.seamlessClone(obj, source_image, mask, center, cv2.MIXED_CLONE)
            except:
                continue
 
# Write results
#cv2.imwrite(source_path + "normal_merge2.jpg", normal_clone)
    #cv2.imwrite(source_path + "fluid_merge"+str(i)+'.jpg', mixed_clone)
            filename,extension=os.path.splitext(sample)

            cv2.imwrite(target_path +'abcd54_and_'+filename+str(j+1)+'.jpg', mixed_clone)
    #print(len(list))
    





