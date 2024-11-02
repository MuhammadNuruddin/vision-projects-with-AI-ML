import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.keyboard import Controller, Key

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, 
                                         max_num_hands=self.max_hands, 
                                         min_detection_confidence=self.detection_conf, 
                                         min_tracking_confidence=self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
        return lm_list

    def fingers_up(self, lm_list):
        if len(lm_list) != 21:
            return []
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        fingers = []

        # Thumb
        if lm_list[tips[0]][1] < lm_list[tips[0] - 2][1]:  # Left of thumb's second joint
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip in tips[1:]:
            if lm_list[tip][2] < lm_list[tip - 2][2]:  # Above the second joint
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def control_media_and_brightness(fingers, keyboard):
    # Open palm: Play/Pause
    if fingers == [0, 1, 1, 1, 1]:  # All fingers except thumb
        keyboard.press(Key.media_play_pause)
        time.sleep(0.5)

    # Index finger only: Volume Up
    elif fingers == [0, 1, 0, 0, 0]:  # Only index finger up
        keyboard.press(Key.media_volume_up)
        time.sleep(0.2)

    # Middle finger only: Volume Down
    elif fingers == [0, 0, 1, 0, 0]:  # Only middle finger up
        keyboard.press(Key.media_volume_down)
        time.sleep(0.2)

    # Ring finger only: Brightness Up
    elif fingers == [0, 0, 0, 1, 0]:  # Only ring finger up
        # Replace with brightness up command
        print("Brightness Up")
        time.sleep(0.2)

    # Pinky only: Brightness Down
    elif fingers == [0, 0, 0, 0, 1]:  # Only pinky finger up
        # Replace with brightness down command
        print("Brightness Down")
        time.sleep(0.2)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_conf=0.7)
    keyboard = Controller()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.get_position(img)

        if lm_list:
            fingers = detector.fingers_up(lm_list)
            control_media_and_brightness(fingers, keyboard)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
