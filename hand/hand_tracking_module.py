import cv2
import mediapipe as mp
import time


class HandLandmark:
    # Define the landmark indices for the finger tips
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


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
        # print(results.multi_hand_landmarks)  
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        return img



    def find_position(self, img, hand_no=0, draw=False):

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)

        return lm_list     

    def fingers_up(self, lm_list):
        """
        Returns a list where each index represents a finger:
        1 if the finger is up, 0 if it's down.
        [Thumb, Index, Middle, Ring, Pinky]
        """
        if len(lm_list) < 21:
            return [0, 0, 0, 0, 0]  # Ensure we have at least 21 landmarks

        fingers = []

        # for the Thumb (Use x-coordinate to check if it's up, assuming horizontal hand orientation)
        if lm_list[HandLandmark.THUMB_TIP][1] > lm_list[HandLandmark.THUMB_IP][1]:  # Check thumb tip vs IP
            fingers.append(1)
        else:
            fingers.append(0)

        # Index finger
        fingers.append(1 if lm_list[HandLandmark.INDEX_FINGER_TIP][2] < lm_list[HandLandmark.INDEX_FINGER_PIP][2] else 0)

        # Middle finger
        fingers.append(1 if lm_list[HandLandmark.MIDDLE_FINGER_TIP][2] < lm_list[HandLandmark.MIDDLE_FINGER_PIP][2] else 0)

        # Ring finger
        fingers.append(1 if lm_list[HandLandmark.RING_FINGER_TIP][2] < lm_list[HandLandmark.RING_FINGER_PIP][2] else 0)

        # Pinky finger
        fingers.append(1 if lm_list[HandLandmark.PINKY_TIP][2] < lm_list[HandLandmark.PINKY_PIP][2] else 0)

        return fingers   

def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
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



if __name__ == "__main__":
    main()