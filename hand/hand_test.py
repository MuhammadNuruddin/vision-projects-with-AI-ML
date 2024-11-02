import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm


prev_time = 0
curr_time = 0
cap = cv2.VideoCapture(1)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    # if len(lm_list) != 0:
    #     print(lm_list[8])
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (250, 0, 250), 3)

    cv2.imshow("Image Output", img)
    cv2.waitKey(1)