import cv2
import time
import os
import hand_tracking_module as htm

# Set camera dimensions
w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(1)  # Adjust the index if necessary
cap.set(3, w_cam)
cap.set(4, h_cam)

# Initialize hand detector
prev_time = 0
detector = htm.HandDetector(detection_conf=0.75)
tip_ids = [4, 8, 12, 16, 20]

# Define a dictionary for ASL gestures based on specific finger configurations
asl_gestures = {
    (0, 0, 0, 0, 0): "Closed fist (Stop)",
    (1, 0, 0, 0, 0): "Thumb (A)",
    (0, 1, 0, 0, 0): "Index finger (B)",
    (1, 1, 0, 0, 0): "Thumb and Index (C)",
    (0, 0, 1, 0, 0): "Middle finger (D)",
    (0, 0, 0, 1, 0): "Ring finger (E)",
    (0, 0, 0, 0, 1): "Pinky finger (F)",
    (1, 1, 1, 0, 0): "Thumb, Index, and Middle (G)",
    (1, 1, 0, 1, 0): "Thumb, Index, and Ring (H)",
    (1, 1, 0, 0, 1): "Thumb, Index, and Pinky (I)",
    (1, 1, 1, 1, 0): "Thumb, Index, Middle, and Ring (J)",
    (1, 1, 1, 1, 1): "All fingers up (Open hand)",
    (0, 1, 1, 1, 1): "Peace sign (V)",
    (0, 1, 1, 0, 0): "Deuces - Peace sign",
    # TO-DO: ADD MORE SIGNS/GESTURES
}

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        fingers = []
        
        # Check if thumb is up
        fingers.append(1 if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1] else 0)

        # Check other fingers
        for id in range(1, 5):  # Check index, middle, ring, and pinky
            fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)

        # Create a tuple of the fingers' states
        finger_tuple = tuple(fingers)

        # Get the corresponding gesture text
        gesture_text = asl_gestures.get(finger_tuple, "Unknown gesture")
        
        # Display gesture meaning on the screen at the top left corner
        cv2.putText(img, gesture_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(img, f'FPS: {int(fps)}', (550, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Show the image output
    cv2.imshow("Image Output", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
