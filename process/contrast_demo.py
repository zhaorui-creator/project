import cv2   #导入openCV包
import numpy as np
import os
import random
#center_list=[(100,100),(200,100),(300,100),(400,100),
            #(100,300),(200,300),(300,300),(400,300),
            #(100,400),(200,400),(300,400),(400,400)]

import random
import itertools
from functools import reduce
import operator
import math
def generate_center():
    random_list=list(itertools.product(range(0,1920),range(0,1080)))
    #print(random_list)
    pointnum_list=[4,5,6,7]
    point_num=random.choice(pointnum_list)
    coords = random.sample(random_list,point_num)
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    org_point = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    org_point=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    org_point.reverse()
    center_list=[]
    for i in org_point:
        i=list(i)
        center_list.append(i)
    return center_list



bound_centerlist = [[(25, 25), (175, 25), (175, 175), (25, 175)],
                    [(125, 25), (275, 25), (275, 175), (125, 175)],
                    [(225, 25), (375, 25), (375, 175), (225, 175)],
                    [(325, 25), (475, 25), (475, 175), (325, 175)],
                    [(25, 125), (175, 125), (175, 275), (25, 275)],
                    [(125, 125), (275, 125), (275, 275), (125, 275)],
                    [(225, 125), (375, 125), (375, 275), (225, 275)],
                    [(325, 125), (475, 125), (475, 275), (325, 275)],
                    [(25, 225), (175, 225), (175, 375), (25, 375)],
                    [(125, 225), (275, 225), (275, 375), (125, 375)],
                    [(225, 225), (375, 225), (375, 375), (225, 375)],
                    [(325, 225), (475, 225), (475, 375), (325, 375)],
                    [(25, 325), (175, 325), (175, 475), (25, 475)],
                    [(125, 325), (275, 325), (275, 475), (125, 475)],
                    [(225, 325), (375, 325), (375, 475), (225, 475)],
                    [(325, 325), (475, 325), (475, 475), (325, 475)],
                    [(30, 45),(100, 15),(100, 15),(330, 240),(50, 250)],
                    [(30, 50),(100, 15),(100, 15),(330, 240),(50, 250)],
                    [(30, 45),(100, 35),(100, 15),(330, 240),(50, 250)],
                    [(30, 45),(100, 15),(100, 45),(330, 240),(50, 250)],
                    [(30, 45),(100, 15),(100, 15),(330, 260),(50, 250)],
                    [(30, 45),(100, 15),(100, 15),(330, 240),(50, 300)],
                    [(25, 45),(100, 15),(100, 15),(330, 240),(50, 250)],
                    [(30, 45),(80, 15),(100, 15),(330, 240),(50, 250)],
                    [(30, 45),(100, 15),(70, 15),(330, 240),(50, 250)],
                    [(30, 45),(100, 15),(100, 15),(300, 240),(50, 250)],
                    [(30, 45),(100, 15),(100, 15),(330, 240),(100, 250)],
                    [(300,300),(450,400),(430,460),(260,430),(200,420)],
                    [(300,350),(450,400),(430,460),(260,430),(200,420)],
                    [(300,300),(450,450),(430,460),(260,430),(200,420)],
                    [(300,300),(450,400),(430,470),(260,430),(200,420)],
                    [(300,300),(450,400),(430,460),(260,430),(200,430)],
                    [(250,150),(335,200),(335,300),(250,350),(165,300),(165,200)]]

light_list=[100,80,60,40,20]
def contrast_brightness_demo(image, c, b):  # C 是对比度，b 是亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)   #改变像素的API
    return dst



def lotadd():
    for i in range(1,29):
        if i<10:
            source_path='/home/dl/Documents/project/classfication/obj_and_background/'+'abcd'+str(0)+str(i)+'/'
            target_path='/home/dl/Documents/project/classfication/obj_and_background+aug/'+'augresult_abcd'+str(0)+str(i)+'/'
        else:
            source_path='/home/dl/Documents/project/classfication/obj_and_background/'+'abcd'+str(i)+'/'
            target_path='/home/dl/Documents/project/classfication/obj_and_background+aug/'+'augresult_abcd'+str(i)+'/'






        print("--------generate------------")
        image_list=sorted(os.listdir(source_path))
        print(image_list)
        for file in image_list:
            lsPointsChoose=random.choice(bound_centerlist)
            print(lsPointsChoose)
            sample_light=random.choice(light_list)
            img=cv2.imread(source_path+file)
            mask=np.zeros(img.shape,np.uint8)
            pts=np.array([lsPointsChoose],np.int32)
            print(pts.shape)

            pts=pts.reshape((-1,1,2))

            #画多边形
            mask=cv2.polylines(mask,[pts],True,(0,255,255))

            #填充多边形
            mask2=cv2.fillPoly(mask,[pts],(255,255,255))

            #颜色反转 得到框外白色  框内黑色的眼膜图像
            height,width,depth=mask2.shape
            mask3=np.zeros((height,width,depth),np.uint8)

            for i in range(0,height):
                for j in range(0,width):
                    (b,g,r)=mask2[i,j]
                    mask3[i,j]=(255-b,255-g,255-r)
            
            #得到roi部分是黑色，区域部分是原图的图像
            source1=cv2.bitwise_and(mask3,img)
            #得到roi部分有图像，框外部分全是黑色的图像
            roi=cv2.bitwise_and(mask2,img)

            #得到经过对比度增强的改变亮度的roi部分
            lightroi=contrast_brightness_demo(roi,1.2,sample_light)
            #将框外经过对比度增强的部分去掉变成全黑
            roiresult=cv2.bitwise_and(mask2,lightroi)

            #得到最终的增强结果
            result=cv2.add(source1,roiresult)
            cv2.imwrite(target_path+'aug'+file,result)
            #roi=src[sample_center[1]-75:sample_center[1]+75,sample_center[0]-75:sample_center[0]+75]
            #image=contrast_brightness_demo(roi,1.2,sample_light)
            #src[sample_center[1]-75:sample_center[1]+75,sample_center[0]-75:sample_center[0]+75]=image
            #cv2.imwrite(target_path+'aug'+file,src)
        #src=cv2.imread('/home/dl/Data/org/abcd01.jpg')  #读取F:/shiyan/1.png路径下的名为1格式为.png的图片
        #image = contrast_brightness_demo(src, 1.2, sample_light)
        #cv2.imwrite('/home/dl/'+'1.jpg',image)
def singleadd():
    #for i in range(1,29):
        #if i<10:
    source_path='/home/dl/Documents/project/classfication/testimg/org/'
    target_path='/home/dl/Documents/project/classfication/2/'
        #else:
            #source_path='/home/dl/Documents/project/classfication/obj_and_background/'+'abcd'+str(i)+'/'
            #target_path='/home/dl/Documents/project/classfication/obj_and_background+aug/'+'augresult_abcd'+str(i)+'/'






    print("--------generate------------")
    image_list=sorted(os.listdir(source_path))
    print(image_list)
    k=1
    for k in range(0,20):
        for file in image_list:
            lsPointsChoose=generate_center()
            print(lsPointsChoose)
            sample_light=random.choice(light_list)
            img=cv2.imread(source_path+file)
            mask=np.zeros(img.shape,np.uint8)
            pts=np.array([lsPointsChoose],np.int32)
            print(pts.shape)

            pts=pts.reshape((-1,1,2))

            #画多边形
            mask=cv2.polylines(mask,[pts],True,(0,255,255))

            #填充多边形
            mask2=cv2.fillPoly(mask,[pts],(255,255,255))

            #颜色反转 得到框外白色  框内黑色的眼膜图像
            height,width,depth=mask2.shape
            mask3=np.zeros((height,width,depth),np.uint8)

            for i in range(0,height):
                for j in range(0,width):
                    (b,g,r)=mask2[i,j]
                    mask3[i,j]=(255-b,255-g,255-r)
            
            #得到roi部分是黑色，区域部分是原图的图像
            source1=cv2.bitwise_and(mask3,img)
            #得到roi部分有图像，框外部分全是黑色的图像
            roi=cv2.bitwise_and(mask2,img)

            #得到经过对比度增强的改变亮度的roi部分
            lightroi=contrast_brightness_demo(roi,1.2,sample_light)
            #将框外经过对比度增强的部分去掉变成全黑
            roiresult=cv2.bitwise_and(mask2,lightroi)

            #得到最终的增强结果
            result=cv2.add(source1,roiresult)
            cv2.imwrite(target_path+'aug'+str(k)+file,result)
            k=k+1
            #roi=src[sample_center[1]-75:sample_center[1]+75,sample_center[0]-75:sample_center[0]+75]
            #image=contrast_brightness_demo(roi,1.2,sample_light)
            #src[sample_center[1]-75:sample_center[1]+75,sample_center[0]-75:sample_center[0]+75]=image
            #cv2.imwrite(target_path+'aug'+file,src)
        #src=cv2.imread('/home/dl/Data/org/abcd01.jpg')  #读取F:/shiyan/1.png路径下的名为1格式为.png的图片
        #image = contrast_brightness_demo(src, 1.2, sample_light)
        #cv2.imwrite('/home/dl/'+'1.jpg',image)
if __name__=='__main__':
    singleadd()
    