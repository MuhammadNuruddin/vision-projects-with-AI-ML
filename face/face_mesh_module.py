import cv2
import mediapipe as mp
import time
import math


class FaceMeshDetector():
    def __init__(self, static_mode=False, max_faces=2, min_detection_conf=0.5, min_track_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf
        self.p_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_mode,
            max_num_faces=self.max_faces,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_track_conf
        )
        # Set drawing specifications
        self.landmark_spec = self.p_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_spec = self.p_draw.DrawingSpec(color=(0, 255, 0))


    def find_face_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.p_draw.draw_landmarks(
                        img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS, 
                        self.landmark_spec, self.connection_spec
                    )

                face = []
                for lm in face_lms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])  
                faces.append(face)

        return img, faces

    def get_distance(self, point1, point2):
        if len(point1) >= 2 and len(point2) >= 2:
            x1, y1 = point1[0], point1[1]
            x2, y2 = point2[0], point2[1]
            return math.hypot(x2 - x1, y2 - y1)
        return None  # Return None if points are not valid

    def detect_eye_blink(self, face):
        if len(face) > 159:
            left_eye_top = face[159]
            left_eye_bottom = face[145]
            distance = self.get_distance(left_eye_top, left_eye_bottom)
            if distance is not None:
                blink_threshold = 5
                return distance < blink_threshold
        return False

    def detect_mouth_open(self, face):
        if len(face) > 14:
            mouth_top = face[13]
            mouth_bottom = face[14]
            distance = self.get_distance(mouth_top, mouth_bottom)
            if distance is not None:
                mouth_open_threshold = 15
                return distance > mouth_open_threshold
        return False


def main():
    file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/face/face_videos/1.mp4'
    cap = cv2.VideoCapture(file_path)

    display_width = 800
    display_height = 500
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
    prev_time = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (display_width, display_height))
        img, faces = detector.find_face_mesh(img)

        if faces:
            for face in faces:
                if detector.detect_eye_blink(face):
                    cv2.putText(img, "Blink Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if detector.detect_mouth_open(face):
                    cv2.putText(img, "Mouth Open", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image Output", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
