import os
import shutil

img_list=os.listdir('/home/luo/rui/background/')
img_list.sort(key=lambda x:int(x.split('.')[0]))

for i in range(0,5347):
    filename = img_list[i]
    file_path = os.path.join('/home/luo/rui/background/'+filename)
    shutil.copy(file_path, '/home/luo/rui/background1/')