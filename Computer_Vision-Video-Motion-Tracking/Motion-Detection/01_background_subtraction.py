import cv2
import numpy as np

# STEP 1: LOAD VIDEO SOURCE
# Initializing the video capture from a local file.
cap = cv2.VideoCapture("../walking.mp4")

# STEP 2: INITIALIZE BACKGROUND SUBTRACTOR (MOG2)
# -------------------------------------------------------------------------
# LOGIC: Here we initialize the "Memory" of the algorithm.
# It builds a model of the static background over time.
# 'detectShadows=True' helps the model distinguish between objects and shadows.
# MOG2 is robust against small background movements like swaying trees.
subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    # Read frame by frame
    ret, frame = cap.read()
    if not ret:
        print("End of video or file not found.")
        break

    # STEP 3: APPLY THE MASK (.apply)
    # -------------------------------------------------------------------------
    # LOGIC: For every new frame, the algorithm compares it to its learned model.
    # - If it detects a moving object: It marks it WHITE (255).
    # - If it detects a shadow: It marks it GRAY (127).
    # - Static background elements: They remain BLACK (0).
    mask = subtractor.apply(frame)

    # STEP 4: VISUALIZATION
    # Displaying both the original feed and the extracted motion mask.
    cv2.imshow("Original Video Feed", frame)
    cv2.imshow("Motion Mask (MOG2 Output)", mask)

    # STEP 5: EXIT LOGIC
    # Press 'q' on the keyboard to exit the loop and close windows.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# CLEANUP: Release resources and close all GUI windows
cap.release()
cv2.destroyAllWindows()