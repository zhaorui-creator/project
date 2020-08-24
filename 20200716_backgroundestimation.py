import numpy as np
import cv2
from skimage import data, filters

#cap = cv2.VideoCapture('/home/dl/data/subway/hongqiao.mp4')

#videoinput1 = cv2.VideoCapture("/home/dl/data/滞留物/hiv00009.mp4")
videoinput1 = cv2.VideoCapture('/home/luo/rui/2.mp4')

counter = 1

initial_frames = []

initialized_flag = False

frames = 0
while(True):
    
    ret1, img1 = videoinput1.read()#(960, 1280, 3) #720
    
    #black_img = np.zeros((960,1280,3))
    #density_img = np.zeros((723,500,3))
    if img1 is None:
        break
    
    #img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    
    if len(initial_frames) < 400:
        if counter % 10 == 0:
            initial_frames.append(img1)
            frames += 1
    elif initialized_flag == False: # 确保初始化只执行一次
        #print("run to here 1")
        #frames = np.concatenate(initial_frames)
        sequence = np.stack(initial_frames, axis=3)
        
        print(sequence.shape)
        medianFrame = np.median(sequence, axis=3)#.astype(dtype=np.uint8) 
        
        background = medianFrame.astype(dtype=np.uint8)
        
        initialized_flag = True # 确保初始化只做一次
        
    #print(len(initial_frames))
    #如果还没有初始化，下面的不用处理
    #print(initialized_flag)
    if initialized_flag == True:
        #更新背景
        #print("run to here 2")
        #if counter % 20 == 0:
        if counter % 1 == 0:
            
           
            medianFrame = 0.98*medianFrame + 0.02*img1
            #medianFrame = 0.9*medianFrame+0.1*img1
            
            #background = medianFrame.astype(dtype=np.uint8) 
            
            #initial_frames.pop(0)
            #initial_frames.append(img1)
            #sequence = np.stack(initial_frames, axis=2)
            #medianFrame = np.median(sequence, axis=2)#.astype(dtype=np.uint8) 
            background = medianFrame.astype(dtype=np.uint8)
            #cv2.imwrite('/home/luo/rui/frames/'+str(counter)+'.jpg',background)
        #cv2.imwrite('/home/luo/rui/background/'+str(counter-4000)+'.jpg',background)
            
        background_show =  cv2.resize(background, (int(background.shape[1] / 2), int(background.shape[0] / 2)))
        img1 =  cv2.resize(img1, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)))
        
        #print(background_show.shape,img1.shape)
        
        imgshow = np.concatenate([img1,background_show],axis=1)
        
        cv2.imshow("11",imgshow)
        cv2.waitKey(10)
        
    #else:
    #    print("run to here 3")
    
    counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


"""
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
print(frameIds)


frames = []

for fid in frameIds:

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

    ret, frame = cap.read()

    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   

cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
"""
