import numpy as np
import cv2

import soundfile as sf
import pyaudio as pa

import time

xs = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def callback(in_data, frame_count, time_info, status):
    global xs
    in_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float)
    in_float[in_float > 0.0] /= float(2**15 - 1)
    in_float[in_float <= 0.0] /= float(2**15)
    # xs = np.r_[xs, in_float]
    xs = 10*np.log10(np.mean(np.abs(in_float)))+30
    # print(10*np.log10(np.mean(np.abs(in_float)))+30)
    return (in_data, pa.paContinue)


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    # pyaudio
    p_in = pa.PyAudio()
    print("device num: {0}".format(p_in.get_device_count()))
    for i in range(p_in.get_device_count()):
        print(p_in.get_device_info_by_index(i))
    py_format = p_in.get_format_from_width(2)
    fs = 44100
    channels = 1
    chunk = 1024
    use_device_index = 3

    # 入力ストリームを作成
    in_stream = p_in.open(format=py_format,
                          channels=channels,
                          rate=fs,
                          input=True,
                          frames_per_buffer=chunk,
                          input_device_index=use_device_index,
                          stream_callback=callback)

    in_stream.start_stream()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # r = 231
        # g = 76
        # b = 60
        r = 255
        g = 234
        b = 167

        line_width = 5
        font_size = 5
        x = 100
        # y = np.random.randint(255,500)
        y = int(xs*10) + 200

        # グレースケールに変換
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for (x,y,w,h) in faces:
        #     frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,0,0),-1)
        #     # frame = cv2.circle(frame,(int(x+w/2),int(y+h/2)), 100, (111,111,0), -1)
        #     roi_gray = gray[y:y+h, x:x+w]
        #     roi_color = frame[y:y+h, x:x+w]


        # 文字を追加
        # cv2.putText(frame,'hello!',(x, y), font, font_size,(b,g,r),line_width,cv2.LINE_AA)

        # frame = cv2.circle(frame,(600,400), int(np.abs(xs*10)), (r,g,b), -1)
        print(int(np.abs(xs*10)))
        if int(np.abs(xs*10)) > 60:
            frame = cv2.circle(frame,(900,400), int(np.abs(xs*15)), (b,g,r), -1)
        # frame = cv2.circle(frame,(800,200), int(np.abs(xs*10)), (100), -1)
        # frame = cv2.circle(frame,(600+int(xs)*10,400), 100, (100,100,100), -1)
        # frame = cv2.circle(frame,(500,300+int(xs)*10), int(np.abs(xs)*10), (r,b,g), -1)


        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    in_stream.stop_stream()
    in_stream.close()

    p_in.terminate()