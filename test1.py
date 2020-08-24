from PIL import Image
import cv2
import numpy as np
a = Image.open('/home/luo/rui/classification/Dataset/0/12.jpg').convert('RGB')
print(a.size)
#cv2.COLOR_RGB2BGR
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(img.shape)
cv2.imshow("OpenCV",img)  
cv2.waitKey()
cv2.destroyAllWindows()  