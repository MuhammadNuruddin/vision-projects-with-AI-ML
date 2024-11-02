import cv2
import time
import pose_module as pm


file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/pose/pose_videos/squat_2.mp4'
cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set width and height
display_width = 800
display_height = 500
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
prev_time = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (display_width, display_height))

    img = detector.find_pose(img)
    lm_list = detector.find_position(img)
    if len(lm_list) != 0:
        feedback = detector.check_squat_form(lm_list)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time


    cv2.putText(img, f"X11Y3R: {feedback}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Squats: {detector.squat_count}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Image Output", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()