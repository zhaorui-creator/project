import cv2
import os


def resize_image():
    """
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,调整图像尺寸
    具体而言,此时某个边正好可以等于目标长度,另一边小于等于目标长度
    将缩放后的数据拷贝到画布中心,返回完成缩放
    """
    image_path='/home/dl/Downloads/indoorCVPR_09/roiresize/'
    target_path='/home/dl/Downloads/indoorCVPR_09/roiresize1/'
    img_list=sorted(os.listdir(image_path))
    i=0
    for file in img_list:
        i=i+1
        img = cv2.imread(image_path+file)
        #print(img)
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = (300,300)#inp_dim是需要resize的尺寸（如416*416）
        # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即保证较长的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))
        image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(target_path+file,image)
    
 #将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.,用了双三次插值

if __name__=='__main__':
    resize_image()


