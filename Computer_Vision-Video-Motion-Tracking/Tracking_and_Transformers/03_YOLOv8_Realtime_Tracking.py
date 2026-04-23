# ==============================================================================
# PROJECT: Real-time Object Tracking using YOLOv8 (BotSORT approach)
# FILENAME: 03_YOLOv8_Realtime_Tracking.py
# DESCRIPTION: This script performs real-time person tracking via webcam 
#              using YOLOv8 and the BotSORT algorithm for better accuracy.
# ==============================================================================

# STEP 1: Import necessary libraries
import cv2  # Import OpenCV for displaying the video stream
from ultralytics import YOLO  # Import YOLO for detection and tracking logic

# STEP 2: Load the pre-trained YOLO model
# We use the 'nano' version (yolov8n.pt) to ensure smooth real-time performance
model = YOLO("yolov8n.pt")

# STEP 3: Initialize Tracking with Stream mode
# source=0: Uses the default laptop/computer webcam
# persist=True: Ensures the model remembers objects across frames
# stream=True: Optimized for long-running video to save memory (generator)
# tracker="botsort.yaml": Uses BotSORT algorithm for more stable tracking
# classes=[0]: Filters to track ONLY 'Person'
results = model.track(source=0, persist=True, stream=True, tracker="botsort.yaml", conf=0.4, classes=[0])

# STEP 4: Process the tracking results in a loop
# Since stream=True is used, 'results' is an efficient generator
for r in results:

    # STEP 5: Annotate the frame
    # r.plot() draws the bounding boxes, labels, and tracking IDs automatically
    annotated_frame = r.plot()

    # STEP 6: Display the live output
    # Opens a window showing the webcam feed with tracking visuals
    cv2.imshow("YOLOv8 Tracking Live - Taliye", annotated_frame)

    # STEP 7: Exit condition
    # The cv2.waitKey(1) is crucial to refresh the window and catch key presses
    # Press 'q' on your keyboard to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 8: Cleanup and release resources
# Closes all OpenCV windows and stops the webcam process
cv2.destroyAllWindows()