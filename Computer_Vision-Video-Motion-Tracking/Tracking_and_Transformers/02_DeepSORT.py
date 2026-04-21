# ==============================================================================
# PROJECT: Object Tracking using YOLOv8 (ByteTrack/DeepSORT approach)
# FILENAME: 02_DeepSORT.py
# DESCRIPTION: This script performs real-time person tracking in a video using
#              YOLOv8 and the ByteTrack algorithm to assign unique IDs to people.
# ==============================================================================

# STEP 1: Import necessary libraries
import cv2  # Import OpenCV for video processing and display
from ultralytics import YOLO  # Import YOLO from Ultralytics for detection and tracking

# STEP 2: Load the pre-trained YOLO model
# We are using the 'nano' version (yolov8n.pt) for faster performance
model = YOLO('yolov8n.pt')

# STEP 3: Setup video source
video_path = "../walking.mp4"  # Path to your input video file
cap = cv2.VideoCapture(video_path)  # Initialize OpenCV video capture object

# STEP 4: Process video frames in a loop
while cap.isOpened():  # Loop as long as the video file is open and readable
    success, frame = cap.read()  # Read a single frame from the video

    if success:  # If the frame was captured successfully

        # --- TRACKING LOGIC ---
        # persist=True: Tells the model to remember objects from the previous frame (maintains IDs)
        # tracker="bytetrack.yaml": Uses the ByteTrack algorithm for high-performance tracking
        # conf=0.5: Only tracks objects where detection confidence is above 50%
        # classes=[0]: Filters detections to ONLY track 'Person' (Class 0 in COCO dataset)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, classes=[0])

        # STEP 5: Annotate the frame (Draw boxes and IDs)
        # results[0].plot() automatically draws bounding boxes and tracking IDs on the frame
        annotated_frame = results[0].plot()

        # STEP 6: Display the output
        # Opens a window to show the live tracking visualization
        cv2.imshow("YOLOv8 Person Tracking - GitHub Repo", annotated_frame)

        # STEP 7: Exit condition
        # If the user presses the 'q' key, the loop will break and the program stops
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:  # If no frames are left (end of video), break the loop
        break

# STEP 8: Cleanup and release resources
cap.release()  # Close the video file or webcam stream
cv2.destroyAllWindows()  # Close all open OpenCV windows