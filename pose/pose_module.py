import cv2
import mediapipe as mp
import time
import os
import numpy as np


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PoseLandmark:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


class PoseDetector():
    def __init__(self, mode=False, smooth=True, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth,
                                      min_detection_confidence=self.detection_conf, 
                                      min_tracking_confidence=self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils
        self.squat_count = 0
        self.squat_in_progress = False

    
    
    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results and self.results.pose_landmarks:  # Check if results is not None
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list

    def check_squat_form(self, lm_list):
        if len(lm_list) > 0:
            left_hip = lm_list[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = lm_list[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = lm_list[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

            angle = self.calculate_angle(left_hip, left_knee, left_ankle)

            # Check for squat form
            if angle > 160:  # Standing position
                if self.squat_in_progress:  # If we were in a squat and now standing
                    self.squat_count += 1
                    self.squat_in_progress = False  # Reset squat in progress

                return "Stand up!"  # User is standing

            elif angle < 90:  # Squatting position
                self.squat_in_progress = True  # We are now in a squat
                return "Keep going down!"  # User is squatting

            else:
                return "Good form!"  # Squat is in good form
        return "No pose detected"

    def calculate_angle(self, p1, p2, p3):
        p1 = np.array(p1[1:])  # Hip
        p2 = np.array(p2[1:])  # Knee
        p3 = np.array(p3[1:])  # Ankle
        angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        angle = np.abs(angle) * 180.0 / np.pi
        if angle > 180.0:
            angle = 360 - angle
        return angle




def main():
    file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/pose/pose_videos/16.mp4'
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
    detector = PoseDetector()
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


        # cv2.putText(img, feedback, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # cv2.putText(img, f"Squats: {detector.squat_count}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("Image Output", img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()