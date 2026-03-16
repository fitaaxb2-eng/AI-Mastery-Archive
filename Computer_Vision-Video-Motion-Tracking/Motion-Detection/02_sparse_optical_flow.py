"""
==========================================================================
TITLE: Lucas-Kanade Optical Flow (Motion Tracking)
GOAL: Detect and track specific points moving across video frames.
APPLICATIONS: Object tracking, Video stabilization, and Gesture recognition.
==========================================================================
"""

# STEP 1: IMPORT LIBRARIES
import cv2
import numpy as np

# -------------------------------------------------------------------------
# LOAD VIDEO SOURCE (WEBCAM OR FILE)
# - To use your LIVE WEBCAM: Set 'video_source = 0'
# - To use a VIDEO FILE: Set 'video_source = "path/to/your/video.mp4"'
# -------------------------------------------------------------------------
video_source = "../walking.mp4"
cap = cv2.VideoCapture(video_source)

# STEP 2: SET FEATURE DETECTION PARAMETERS (Shi-Tomasi Corner Detector)
# maxCorners: Max points to track, qualityLevel: Detection threshold, 
# minDistance: Space between points, blockSize: Search area size.
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=10, blockSize=7)

# STEP 3: SET LUCAS-KANADE TRACKING PARAMETERS
# winSize: Search window size, maxLevel: Image pyramids (to catch fast motion),
# criteria: When to stop calculation (after 10 iterations or 0.03 accuracy).
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# STEP 4: INITIALIZE TRACKING
# Capture the first frame and convert to grayscale
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Find the best initial points (corners) to track in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a black "mask" image to draw the movement lines (trails)
mask = np.zeros_like(old_frame)

# STEP 5: START VIDEO PROCESSING LOOP
while True:
    ret, frame = cap.read()  # Read the current frame
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to gray

    # STEP 6: RE-DETECT POINTS IF LOST
    # If points go out of screen or drop below 10, find new points to track
    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)  # Reset the drawing mask

    # STEP 7: CALCULATE OPTICAL FLOW
    if p0 is not None:
        # Calculate new positions (p1) of the points using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # STEP 8: FILTER AND DRAW TRACKED POINTS
        if p1 is not None:
            # Select only points that were successfully tracked (status == 1)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Loop through points to draw lines and circles
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # Current point coordinates (x, y)
                c, d = old.ravel()  # Previous point coordinates (x, y)

                # Draw a green line on the mask (the trail)
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

                # Draw a red dot on the current frame (the object)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            # STEP 9: DISPLAY THE RESULT
            # Combine the frame with the tracking trails
            img = cv2.add(frame, mask)
            cv2.imshow('Lucas Kanade - Motion Tracker', img)

            # STEP 10: UPDATE PREVIOUS FRAME AND POINTS
            old_gray = frame_gray.copy()  # Current gray becomes previous gray
            p0 = good_new.reshape(-1, 1, 2)  # Current points become previous points

    # Display raw frame if no points are tracked
    else:
        cv2.imshow('Lucas Kanade - Motion Tracker', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# STEP 11: CLEANUP
cap.release()
cv2.destroyAllWindows()