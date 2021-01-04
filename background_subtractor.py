import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    dst = cv2.addWeighted(frame[:,:,2],0.1,fgmask,0.9,0)

    # cv2.imshow('frame',dst)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()