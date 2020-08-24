#import cv2
#videoinput1 = cv2.VideoCapture('/home/luo/rui/0413_01.mp4')
#frames=[]
#while True:
    
    #ret1, img1 = videoinput1.read()
    #frames.append(img1)
#print(len(frames))







import cv2

vc = cv2.VideoCapture('/home/luo/rui/1.avi')


while True:
    ret, frame = vc.read()
    if ret == True:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('test', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break


