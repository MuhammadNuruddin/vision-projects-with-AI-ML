import cv2
import time
import pyautogui
import hand_tracking_module as htm
import numpy as np

# Setup webcam capture
cap = cv2.VideoCapture(1)

detector = htm.HandDetector(detection_conf=0.75)

# Variables for screen size and smoothing factor
screen_width, screen_height = pyautogui.size()
smooth_factor = 7  # Adjust this value for smoother movement

# Previous mouse location for smoother transitions
prev_mouse_x, prev_mouse_y = 0, 0

prev_time = 0

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally to correct mirrored effect
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (800, 500))
    img = detector.find_hands(img, draw=True)

    # Get the positions of landmarks
    lm_list = detector.find_position(img)

    if lm_list:
        # Get coordinates of the index finger for mouse movement
        index_finger = lm_list[htm.HandLandmark.INDEX_FINGER_TIP]

        # Scale the coordinates to the screen size and apply smoothing
        x = np.interp(index_finger[1], (0, 800), (0, screen_width))
        y = np.interp(index_finger[2], (0, 500), (0, screen_height))
        mouse_x = prev_mouse_x + (x - prev_mouse_x) / smooth_factor
        mouse_y = prev_mouse_y + (y - prev_mouse_y) / smooth_factor
        pyautogui.moveTo(mouse_x, mouse_y)

        # Update the previous mouse positions
        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Check for clicking gestures
        fingers = detector.fingers_up(lm_list)
        
        # Left Click: Index and middle fingers up, close together
        if fingers == [1, 1, 0, 0, 0]:
            pyautogui.click()

        # Right Click: Index, middle, and ring fingers up
        elif fingers == [1, 1, 1, 0, 0]:
            pyautogui.rightClick()

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Show the image output
    cv2.imshow("Image Output", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
