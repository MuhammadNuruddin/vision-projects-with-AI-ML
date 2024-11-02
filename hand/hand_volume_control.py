import cv2
import numpy as np
import time
import hand_tracking_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



w_cam, h_cam = 640, 480

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()

min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
bar_vol = 400
vol_perc = 0
# (-63.5, 0.0, 0.5)


cap = cv2.VideoCapture(1)
cap.set(3, w_cam)
cap.set(4, h_cam)
prev_time = 0

detector = htm.HandDetector(detection_conf=0.7)


while True:
    succcess, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list[2])
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 8, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 8, (255,0,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
        cv2.circle(img, (cx,cy), 9, (255,255,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2 - y1)
        # print(length)
        # range of fingers distance = 50 - 300
        # volume range = -65 - 0
        vol = np.interp(length, [20,210], [min_vol, max_vol])
        bar_vol = np.interp(length, [20,210], [400, 150]) 
        vol_perc = np.interp(length, [20,210], [0, 100]) 
        volume.SetMasterVolumeLevel(vol, None)
        if length < 50:
            cv2.circle(img, (cx,cy), 9, (0,255,0), cv2.FILLED)
        
    cv2.rectangle(img, (50, 150), (85, 400), (0,255,0), 2)
    cv2.rectangle(img, (50, (int(bar_vol))), (85, 400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_perc)}%', (40,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.imshow("Image Output", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources outside the loop
cap.release()
cv2.destroyAllWindows()