import cv2
import time
import face_mesh_module as fmm

file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/face/face_videos/1.mp4'
cap = cv2.VideoCapture(file_path)

display_width = 800
display_height = 500
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
prev_time = 0
detector = fmm.FaceMeshDetector()

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if there is an issue reading the frame

    img = cv2.resize(img, (display_width, display_height))
    img, faces = detector.find_face_mesh(img)
    # Gesture detection for each face
    if faces:
        for face in faces:
            if detector.detect_eye_blink(face):
                cv2.putText(img, "Blink Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if detector.detect_mouth_open(face):
                cv2.putText(img, "Mouth Open", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources outside the loop
cap.release()
cv2.destroyAllWindows()