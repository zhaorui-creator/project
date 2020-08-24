import os
import random
def select_file(fileDir): 
    pathDir = sorted(os.listdir(fileDir))    #取图片的原始路径
    filenumber=len(pathDir)
    sample=random.choice(pathDir)
    print (sample)
    return sample


if __name__=='__main__':
    for i in range(10):
        sample=select_file('/home/dl/roi1/')
        print(sample)