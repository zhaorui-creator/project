import cv2
import os

import cv2
import os
target_path='/home/dl/Downloads/indoorCVPR_09/images/'

source_path='/home/dl/Downloads/indoorCVPR_09/image/'

sourceimg_list=sorted(os.listdir(source_path))
print(sourceimg_list)
for file in sourceimg_list:
    img_source=cv2.imread(source_path+file)
    try:
        image = cv2.resize(img_source, (1000,1000),0,0, cv2.INTER_LINEAR)
    except:
        continue
    cv2.imwrite(target_path+file,image)
print('complete')