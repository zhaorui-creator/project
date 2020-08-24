import cv2   #导入openCV包
import numpy as np
import os
import random
import random
import itertools
from functools import reduce
import operator
import math
from PIL import Image
from torchvision import transforms



# def generate_center():
#     random_list=list(itertools.product(range(0,500),range(0,1000)))
#     #print(random_list)
#     pointnum_list=[4,5,6,7]
#     point_num=random.choice(pointnum_list)
#     coords = random.sample(random_list,point_num)
#     center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
#     org_point = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
#     org_point=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
#     org_point.reverse()
#     center_list=[]
#     for i in org_point:
#          i=list(i)
#          center_list.append(i)
#     return center_list



# def contrast(img,p):
#     def generate_center():
#     random_list=list(itertools.product(range(0,500),range(0,1000)))
#     #print(random_list)
#     pointnum_list=[4,5,6,7]
#     point_num=random.choice(pointnum_list)
#     coords = random.sample(random_list,point_num)
#     center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
#     org_point = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
#     org_point=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
#     org_point.reverse()
#     center_list=[]
#     for i in org_point:
#          i=list(i)
#          center_list.append(i)
#     return center_list
#     if round(np.random.uniform(0, 1), 1) <= p:
        
#         bound_centerlist = generate_center()
#         lsPointsChoose=random.choice(bound_centerlist)
#         sample_light=random.choice(light_list)
#         #img=cv2.imread(source_path+file)
#         mask=np.zeros((img.size[1],img.size[0],3),np.uint8)
#         pts=np.array([lsPointsChoose],np.int32)
#         pts=pts.reshape((-1,1,2))

#         #画多边形
#         mask=cv2.polylines(mask,[pts],True,(0,255,255))

#         #填充多边形
#         mask2=cv2.fillPoly(mask,[pts],(255,255,255))

#         #颜色反转 得到框外白色  框内黑色的眼膜图像
#         height,width,depth=mask2.shape
#         mask3=np.zeros((height,width,depth),np.uint8)

#         for i in range(0,height):
#             for j in range(0,width):
#                 (b,g,r)=mask2[i,j]
#                 mask3[i,j]=(255-b,255-g,255-r)
        
#         #得到roi部分是黑色，区域部分是原图的图像
#         source1=cv2.bitwise_and(mask3,img)
#         #得到roi部分有图像，框外部分全是黑色的图像
#         roi=cv2.bitwise_and(mask2,img)

#         #得到经过对比度增强的改变亮度的roi部分
#         lightroi=contrast_brightness_demo(roi,1.2,sample_light)
#         #将框外经过对比度增强的部分去掉变成全黑
#         roiresult=cv2.bitwise_and(mask2,lightroi)

#         #得到最终的增强结果
#         result=cv2.add(source1,roiresult)

def generate_center():
    random_list=list(itertools.product(range(0,500),range(0,1000)))
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


def contrast_brightness_demo(image, c, b):  # C 是对比度，b 是亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1-c, b)   #改变像素的API
    return dst

def contrast(img,p):
    
    

    light_list=[100,80,60,40,20]
    #if round(np.random.uniform(0, 1), 1) <= p:
        
    bound_centerlist = generate_center()
    lsPointsChoose=random.choice(bound_centerlist)
    sample_light=random.choice(light_list)
    #img=cv2.imread(source_path+file)
    mask=np.zeros(img.shape,np.uint8)
    pts=np.array([lsPointsChoose],np.int32)
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
    result=Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    return result








def singleadd(img,p):
    light_list=[100,80,60,40,20]
    #for i in range(1,29):
        #if i<10:
    #source_path='/home/dl/Documents/project/classfication/testimg/org/'
    #target_path='/home/dl/Documents/project/classfication/2/'
        #else:
            #source_path='/home/dl/Documents/project/classfication/obj_and_background/'+'abcd'+str(i)+'/'
            #target_path='/home/dl/Documents/project/classfication/obj_and_background+aug/'+'augresult_abcd'+str(i)+'/'
    a=round(np.random.uniform(0, 1), 1) <= p
    print(a)
    if a:
        






        print("--------generate------------")
        #image_list=sorted(os.listdir(source_path))
        #print(image_list)
        #k=1
        #for k in range(0,20):
            #for file in image_list:
        lsPointsChoose=generate_center()
        print(lsPointsChoose)
        sample_light=random.choice(light_list)
        #img=Image.open('/home/dl/rui/project/Dataset/0/1.jpg').convert('RGB')
        #img=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)
        #img=cv2.imread('/home/dl/rui/project/Dataset/0/1.jpg')
        mask=np.zeros(img.shape,np.uint8)
        pts=np.array([lsPointsChoose],np.int32)
        print(pts.shape)

        pts=pts.reshape((-1,1,2))

        #画多边形
        mask=cv2.polylines(mask,[pts],True,(0,255,255))

        #填充多边形
        mask2=cv2.fillPoly(mask,[pts],(255,255,255))

        #颜色反转 得到框外白色  框内黑色的掩膜图像
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
        #cv2.imwrite('1.jpg',result)
                #k=k+1
        cv2.imwrite('/home/luo/rui/argumentation/2/'+file,result)
        


if __name__=='__main__':
    #a = Image.open('/home/dl/rui/project/Dataset/0/1.jpg').convert('RGB')
    #a = cv2.cvtColor(np.asarray(a),cv2.COLOR_BGR2RGB)


    # img=cv2.imread('/home/luo/rui/classification/Dataset/0/1.jpg')
    # result,a=singleadd(img,0.5)
    # if a:
    #     cv2.imwrite('/home/luo/rui/argumentation/0/1.jpg',result)
    # else:
    #     cv2.imwrite('/home/luo/rui/argumentation/0/1.jpg',img)
    # print(a)
    
    #result = contrast(a,0.5)
    #cv2.imwrite('1.jpg',result)

    source_path = '/home/luo/rui/classification/Dataset/2/'
    target_path = '/home/luo/rui/argumentation/2/'

    img_list=os.listdir(source_path)
    img_list.sort(key=lambda x:int(x.split('.')[0]))
    for file in img_list:
        img=cv2.imread(source_path+file)
        #img=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)

        singleadd(img,0.5)
        #print(a)
        #cv2.imwrite('1.jpg',result)
        #result=Image.fromarray(cv2.cvtColor(result,cv2.COLOR_RGB2BGR))

        #if a==True:

        #cv2.imwrite(target_path+file,result)
        #else:
            #cv2.imwrite(target_path+file,img)




    #tf=transforms.Resize((int(200),int(400)))
    #tf=transforms.Compose([
                #transforms.RandomHorizontalFlip(p=0.5),
                #lambda x:Image.open(x).convert('RGB'), #string path =>image data:
                #lambda x:cv2.cvtColor(np.asarray(x),cv2.COLOR_BGR2RGB),
                #transforms.Resize((int(1000*0.5),int(2000*0.5))),
                #lambda x:singleadd(x,0.5),
                #lambda x:Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                #transforms.RandomRotation(15),
                #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.2),
                #lambda x:contrast(x,0.6),
                #transforms.Resize((int(500*1.25),int(1000*1.25))),
                #transforms.CenterCrop((500,1000)),
                #transforms.ToTensor(),
                #transforms.Normalize(normMean,normStd),
                #lambda y:y.view(6,500,500)
                #])
    #result=tf(result)
    #result=result.cuda()
    #print(result.device)
    #result.save('./1.jpg')
    