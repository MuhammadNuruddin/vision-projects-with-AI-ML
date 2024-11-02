# AI Vision Projects

Welcome to my AI Vision Projects repository! Here you'll find a collection of exciting projects that leverage computer vision technologies to perform various tasks, including face detection, workout analysis, and gesture recognition. Each project comes with a video demonstration and usage instructions.

### ⚙️ Features
- **Real-time face detection and tracking** using MediaPipe.
- **Automated alerts** for basic functions like suspicious behavior, intrusion, etc., through face capture for security checks.
- **User-friendly interface** to customize detection parameters.
- **Scalable** for integration into larger security systems or IoT devices.

### 🛠️ Technologies Used
- **OpenCV**: Real-time computer vision library
- **MediaPipe**: Efficient cross-platform machine learning framework
- **Python**: Core programming language

## Project Demos

### Face Detection
![Face Detector](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/91cdaba9-9d55-48f5-86fd-65cd8626685a.gif?ClientID=sulu&Date=1730580970&Signature=884287ca054a54378d472debb682630ec65214bc)
*Demonstrates face detection using an input video or webcam.*

### Face Detection (Camera)
![Face Detector Camera](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/49cc1048-f295-431f-965d-ca9518aaa09f.gif?ClientID=sulu&Date=1730582921&Signature=24768b7ffe535dcd3b29791b1160c129db0be2f1)
*Live camera feed for face detection.*

### Squat Detection
![Squat Detector](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/213496ed-9e35-49d5-91ce-511cb956efc9.gif?ClientID=sulu&Date=1730581214&Signature=85c2ef9a290bb7514760fc8aa8868d0d85dc71d5)
*Tracks and analyzes squat form.*

### Virtual Mouse
![Virtual Mouse](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/398a9004-a484-47b5-af45-f2641c08aadf.gif?ClientID=sulu&Date=1730581403&Signature=b0ec9dfe06c0f50690a2cb9e5dd0ca5a432aa0bd)
*Control your computer with hand gestures!*

### Sign Language Recognition
![Sign Language](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/2f7e113a-718c-459a-bf0b-03faa4587f89.gif?ClientID=sulu&Date=1730581645&Signature=9e5f2746202aa755f1e7ba489832fb4714b7cff7)
*Recognizes and interprets sign language gestures.*

### Pose Tracking Raid
![Pose Tracking Raid](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/4461fa30-8efa-4dfd-9d59-7a02e660864a.gif?ClientID=sulu&Date=1730581798&Signature=d73e67871810e1aa018d3ed52fccad31c7ce9feb)
*Tracks human poses in real-time.*

### Lateral Raise Detection
![Lateral Raise](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/346742cc-23da-480d-ad76-d756ff0622c4.gif?ClientID=sulu&Date=1730581936&Signature=4ceb902025aad9773db1f151fa6cd554730eb10c)
*Analyzes lateral raise exercise form.*

### Lateral Raise (Camera)
![Lateral Raise Camera](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/211d44d9-c25f-4faf-8f1f-23ea2bfc0697.gif?ClientID=sulu&Date=1730582071&Signature=02bd214accff3b852aad26006606b2f63206e0a5)
*Live camera analysis of lateral raises.*

### Hand Volume Control
![Hand Volume Control](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/fd2f7d90-7a43-4e14-8bed-5a085a085f14.gif?ClientID=sulu&Date=1730582279&Signature=94783000d0fe68dec592c53acc39808b86a6425e)
*Control the volume using hand gestures!*

### Hand Drawing
![Hand Draw](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/64574ac0-828a-4ebc-b38f-19c7e02a039e.gif?ClientID=sulu&Date=1730582480&Signature=34ec69f28be20d7518875e0589820d9d8950c279)
*Draw with your hands using computer vision techniques.*

### Hand Counting
![Hand Count](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/9202e4fd-c60b-4eb2-8151-291001f0964b.gif?ClientID=sulu&Date=1730582606&Signature=486787f351c16097ec2a4a114e53eb80bcfe6ee7)
*Counts hand gestures for various applications.*

### Face Mesh Expression
![Face Mesh Expression](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/219a9791-0b61-4de4-af6e-4d949635a0a0.gif?ClientID=sulu&Date=1730582755&Signature=3ee5d655691f324aa1a921b869d126c675fe9e3c)
*Analyzes facial expressions using mesh techniques.*


### 🚀 Getting Started

Follow these steps to get this project up and running on your local machine!

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MuhammadNuruddin/vision-projects-with-AI-ML.git
   cd your-project-repo


2. Set Up a Virtual Environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate 


3. Install Dependencies Make sure you have requirements.txt in your project root. Then run:
   pip install -r requirements.txt


🎥 Usage Instructions:
You can switch between using a camera feed or video files for your project.

- Using the Webcam: Ensure your webcam is connected and modify the code to initialize the camera. Typically, this involves setting the video source to 0(or whatever your camera number is if you're connecting multiple to your PC) in your code:
    cap = cv2.VideoCapture(0)  # Use the default camera

- Using a Video File: To use a video file, specify the path to the video file in your code:
    cap = cv2.VideoCapture('path/to/your/video.mp4')  # Specify your video file path

Make sure to replace 'path/to/your/video.mp4' with the actual path to your video file.

🤝 Contributions
Contributions are welcome! If you want to contribute to this project, please create a new branch, make your changes, and submit a pull request.