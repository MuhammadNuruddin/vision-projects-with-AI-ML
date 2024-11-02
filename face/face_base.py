import cv2
import mediapipe as mp
import time



file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/face/face_videos/4.mp4'
cap = cv2.VideoCapture(file_path)


mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.75)

display_width = 800
display_height = 500
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
prev_time = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (display_width, display_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.location_data.relative_bounding_box)
            # mp_draw.draw_detection(img, detection)
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                   int(bbox_c.width * iw), int(bbox_c.height * ih)
            cv2.rectangle(img, bbox, (0,255,0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

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