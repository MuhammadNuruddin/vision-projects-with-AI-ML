import cv2
import time
import pose_module as pm

# Define the path to your video file or use 0 for webcam
file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/pose/pose_videos/exercise/7.mp4'  # Change to 0 for webcam
cap = cv2.VideoCapture(file_path)

detector = pm.PoseDetector(detection_conf=0.75)

prev_time = 0
count = 0
direction = None  # To track the direction of the movement

while True:
    success, img = cap.read()
    if not success:
        break  # Break the loop if the video ends or there are no frames

    img = cv2.resize(img, (800, 500))
    img = detector.find_pose(img, draw=False)

    # Get positions of landmarks
    lm_list = detector.find_position(img, draw=False)

    if lm_list:
        # Get positions of the left and right shoulders and wrists
        left_shoulder = lm_list[pm.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm_list[pm.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = lm_list[pm.PoseLandmark.LEFT_WRIST]
        right_wrist = lm_list[pm.PoseLandmark.RIGHT_WRIST]

        # Check for lateral raises
        if left_wrist[2] < left_shoulder[2] and right_wrist[2] < right_shoulder[2]:  # Check y-coordinates
            if direction != "up":
                direction = "up"
                count += 1  # Increment count when both wrists go up
        elif left_wrist[2] > left_shoulder[2] and right_wrist[2] > right_shoulder[2]:
            if direction != "down":
                direction = "down"

        # Draw circles around shoulders
        cv2.circle(img, (left_shoulder[1], left_shoulder[2]), 8, (0, 0, 255), cv2.FILLED)  # Red circle for left shoulder
        cv2.circle(img, (right_shoulder[1], right_shoulder[2]), 8, (0, 0, 255), cv2.FILLED)  # Red circle for right shoulder
        cv2.circle(img, (left_shoulder[1], left_shoulder[2]), 15, (0, 0, 255), 2)  # Larger red circle for left shoulder
        cv2.circle(img, (right_shoulder[1], right_shoulder[2]), 15, (0, 0, 255), 2)  # Larger red circle for right shoulder

        # Draw count on image
        cv2.putText(img, f'Count: {count}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Draw vertical bar for count
        bar_height = 200  # Height of the bar
        bar_width = 30  # Width of the bar
        bar_x = 20  # X position of the bar
        bar_y = 400  # Y position (bottom) of the bar

        # Progress effect for the count bar
        current_bar_height = int(bar_height * (count % 10) / 10)  # Normalize the count to the bar height
        cv2.rectangle(img, (bar_x, bar_y - current_bar_height), (bar_x + bar_width, bar_y), (0, 255, 0), -1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS below the count
    cv2.putText(img, f'FPS: {int(fps)}', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Show the image output
    cv2.imshow("Image Output", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
