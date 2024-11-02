import cv2
import time
import os
import hand_tracking_module as htm

w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, w_cam)
cap.set(4, h_cam)

overlay_size = (170, 170)

folder_path = "C:/Users/User/Desktop/X11Y3R/python projects/hand/fingers"
my_list = os.listdir(folder_path)
# print(my_list)
overlay_list = []
for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    # Resize image to the target size
    resized_image = cv2.resize(image, overlay_size)
    overlay_list.append(resized_image)

prev_time = 0
detector = htm.HandDetector(detection_conf=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        fingers = []
        # thumb, right hand
        if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1,5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_fingers = fingers.count(1)

        img[0:170, 0:170] = overlay_list[total_fingers - 1]

        cv2.rectangle(img, (20,225), (170, 425), (0, 255,0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,255,255), 25)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f'FPS: {int(fps)}', (450,60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources outside the loop
cap.release()
cv2.destroyAllWindows()
