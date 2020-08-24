import os

orgimglist=os.listdir('/home/dl/Downloads/indoorCVPR_09/images1/')
objimglist=os.listdir('/home/dl/Downloads/indoorCVPR_09/background_and_obj/')

orgimglist.sort(key=lambda x: int(x.split('.')[0]))
objimglist.sort(key=lambda x: int(x.split('_')[0]))
print(len(objimglist))
#for file in objimglist:
    #print(file)

#for file in orgimglist:
    #print(file.split('.')[0][5:])
#for file in objimglist:
    #filename=file.split('_')[0]
    #print(file)
    #print('---------------')
    #print(filename)

 


