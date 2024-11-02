import cv2
import mediapipe as mp
import time


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/pose/pose_videos/1.mp4'
cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set width and height
display_width = 800
display_height = 500
prev_time = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (display_width, display_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0,0), cv2.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time


    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image Output", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()