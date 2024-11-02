import cv2
import mediapipe as mp
import time
import os


class FaceDetector():
    def __init__(self, min_detection_conf = 0.5,capture_interval=15):
        self.min_detection_conf = min_detection_conf
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_conf)
        self.capture_interval = capture_interval  # Capture interval in seconds
        self.last_capture_time = 0  # Initialize last capture time


    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                       int(bbox_c.width * iw), int(bbox_c.height * ih)
                
                bboxs.append([bbox, detection.score])
                if draw:
                    img = self.corner_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                                (bbox[0], bbox[1] - 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                    cv2.putText(img, "Target Locked", (bbox[0], bbox[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Capture face if interval has passed
                if time.time() - self.last_capture_time > self.capture_interval:
                    self.capture_face(img, bbox)
                    self.last_capture_time = time.time()
        return img, bboxs


    def corner_draw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw bounding box with corner lines
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        
        # Draw sniper cross "+" in the center of the bounding box
        center_x, center_y = x + w // 2, y + h // 2
        cross_length = 15
        cv2.line(img, (center_x - cross_length, center_y), 
                (center_x + cross_length, center_y), (0, 0, 255), t)
        cv2.line(img, (center_x, center_y - cross_length), 
                (center_x, center_y + cross_length), (0, 0, 255), t)

        return img


    def capture_face(self, img, bbox):
        x, y, w, h = bbox
        face_img = img[y:y+h, x:x+w]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = f"detected_faces/face_{timestamp}.jpg"
        if not os.path.exists("detected_faces"):
            os.makedirs("detected_faces")
        cv2.imwrite(save_path, face_img)
        print(f"Face captured and saved to {save_path}")

    # def corner_draw(self, img, bbox, l = 30, t=5, rt=1):
    #     x, y, w, h = bbox
    #     x1, y1 = x+w, y+h
    #     cv2.rectangle(img, bbox, (0,255,0), rt)
    #     # top left x, y
    #     cv2.line(img, (x, y), (x+l, y), (0,255,0), t)
    #     cv2.line(img, (x, y), (x, y+l), (0,255,0), t)

    #     # top right x1, y
    #     cv2.line(img, (x1, y), (x1-l, y), (0,255,0), t)
    #     cv2.line(img, (x1, y), (x1, y+l), (0,255,0), t)

    #     # bottom left x1, y1
    #     cv2.line(img, (x1, y1), (x1+l, y1), (0,255,0), t)
    #     cv2.line(img, (x1, y1), (x1, y1-l), (0,255,0), t)

    #     # bottom right x1, y1
    #     cv2.line(img, (x1, y1), (x1-l, y1), (0,255,0), t)
    #     cv2.line(img, (x1, y1), (x1, y1-l), (0,255,0), t)

    #     return img

def main():
    file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/face/face_videos/1.mp4'
    cap = cv2.VideoCapture(file_path)
    display_width = 800
    display_height = 500
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
    prev_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (display_width, display_height))
        img, bboxs = detector.find_faces(img)
        curr_time = time.time()

        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Output Image", img)


        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()