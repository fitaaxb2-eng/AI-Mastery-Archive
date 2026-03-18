"""
MOTION DETECTION VIA FRAME DIFFERENCING
=======================================
This script detects real-time movement by comparing consecutive frames from a 
video source. It calculates the absolute difference between pixels, filters 
out background noise using a binary threshold, and highlights moving objects 
in white. This lightweight method is ideal for surveillance triggers, wildlife 
monitoring, and low-power IoT applications.

Works with: Live Webcams (0) and Recorded Video Files (.mp4, .avi, etc.)
"""

# ======================================================================================
# Step 1: Importing Libraries
# ======================================================================================
import cv2
import numpy as np

# ======================================================================================
# Step 2: Loading Video Source (Webcam or File)
# ======================================================================================
# Use 0 for live webcam or "filename.mp4" for a specific video file
video_source = 0
cap = cv2.VideoCapture(video_source)

# Step 3: Initialize the first frame
# We read the first frame to act as the initial background reference.
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    # Step 4: Capture the next frame (Current Frame)
    ret, frame2 = cap.read()
    if not ret:
        break  # Exit if the video ends or camera is disconnected

    # Step 5: Convert current frame to Grayscale
    # Simplifies processing by focusing on light intensity rather than color.
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Step 6: Calculate the Absolute Difference
    # This identifies pixels that changed between the previous and current frame.
    diff = cv2.absdiff(gray1, gray2)

    # Step 7: Apply Binary Thresholding
    # - _: Ignored variable (threshold value).
    # - 30: Sensitivity limit. Changes smaller than 30 pixels are ignored (Black).
    # - 255: Motion pixels are turned into Pure White.
    # - THRESH_BINARY: Ensures the output is strictly Black & White.
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Step 8: Visualize the result
    # Shows a window where motion is represented by white shapes.
    cv2.imshow("Motion Detection Output", thresh)

    # Step 9: Update the reference frame
    # Current frame becomes the "previous" frame for the next loop iteration.
    gray1 = gray2

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Step 10: Cleanup
# Release the hardware and close all active windows.
cap.release()
cv2.destroyAllWindows()

# ============================================================
# TECHNICAL SUMMARY:
# ============================================================
# 1. Image Difference (absdiff): "Did something move right now?"
# 2. Threshold (30): "Is the movement big enough to care about?"
# 3. Binary Output: "Show me ONLY the motion, hide the background."
# ============================================================