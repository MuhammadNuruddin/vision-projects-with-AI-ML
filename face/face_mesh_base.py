import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
# Set drawing specifications with color as green (0, 255, 0)
landmark_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
connection_spec = mp_draw.DrawingSpec(color=(0, 255, 0))

file_path = 'C:/Users/User/Desktop/X11Y3R/python projects/face/face_videos/1.mp4'
cap = cv2.VideoCapture(file_path)

display_width = 800
display_height = 500
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if there is an issue reading the frame

    img = cv2.resize(img, (display_width, display_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms, mp_face_mesh.FACEMESH_CONTOURS, landmark_spec, connection_spec)
            for id,lm in enumerate(face_lms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x * iw), int(lm.y * ih)


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