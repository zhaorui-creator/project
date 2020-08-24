import cv2
import os

out_path = "/home/luo/rui/"
def saveFile(file_path):
    cap = cv2.VideoCapture(file_path)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(os.path.join(out_path, '1.avi'), fourcc, fps, size)

    size_scale = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) /4))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_index += 1
            print(frame_index)
            frame = cv2.resize(frame, size_scale)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("it's over")
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':

    saveFile('/home/luo/rui/hiv00009.mp4')