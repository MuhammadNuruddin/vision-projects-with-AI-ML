import cv2
import time
import face_module as fm


cap = cv2.VideoCapture(0)
detector = fm.FaceDetector(min_detection_conf=0.75)
while True:
    success, img = cap.read()
    if not success:
        break
    img, bboxs = detector.find_faces(img)
    cv2.imshow("Face Detector", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()