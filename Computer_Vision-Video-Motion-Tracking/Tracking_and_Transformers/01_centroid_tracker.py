# ==========================================
# CENTROID TRACKING SYSTEM
# Filename: 01_centroid_tracker.py
# Description: This script implements a simple Centroid
# Tracking algorithm using OpenCV. It detects human faces,
# calculates their center point (centroid), and tracks their movement across
# the screen by drawing a visual trail.
# ==========================================

# ==========================================
# STEP 1: IMPORTING LIBRARIES
# ==========================================
import cv2

# ==========================================
# STEP 2: LOADING THE MODEL & CAMERA SETUP
# ==========================================
# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam (ID 0 is the default camera)
cap = cv2.VideoCapture(0)

# Variables to store the previous frame's center point (Memory)
old_cx, old_cy = 0, 0

print("System Active: Tracking faces... Press 'q' to quit.")

# ==========================================
# STEP 3: MAIN PROCESSING LOOP
# ==========================================
while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to Grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # ==========================================
    # STEP 4: CENTROID CALCULATION & VISUALS
    # ==========================================
    for i, (x, y, w, h) in enumerate(faces):
        # Calculate the Centroid (Middle of the face)
        # cx = start point (x) + half the width (w/2)
        cx = int(x + (w / 2))
        cy = int(y + (h / 2))

        # 1. Draw a Blue Box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 2. Draw a Green Dot at the center (Centroid)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 3. Add Text: ID of the face and its pixel coordinates
        cv2.putText(frame, f"ID: {i + 1} | Pos: {cx},{cy}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==========================================
        # STEP 5: TRACKING LOGIC (MOVEMENT TRAIL)
        # ==========================================
        # Check if we have a previous position to compare with
        if old_cx != 0 and old_cy != 0:
            # Euclidean Distance Math: measures the gap between old and new points
            distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5

            # Only track if the movement is within a reasonable range (100 pixels)
            if distance < 100:
                # Draw a Red line showing the path of movement
                cv2.line(frame, (old_cx, old_cy), (cx, cy), (0, 0, 255), 2)

        # Update the memory for the next frame
        old_cx, old_cy = cx, cy

    # Display the live video tracking window
    cv2.imshow('Day 22: Simple Centroid Tracking', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# STEP 6: RELEASING RESOURCES
# ==========================================
cap.release()
cv2.destroyAllWindows()