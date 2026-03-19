"""
AI OBJECT DETECTION & MOTION TRACKING (MobileNet-SSD)
====================================================
This script combines Background Subtraction (MOG2) and Deep Learning (MobileNet-SSD)
to detect and identify moving objects. It filters background noise, detects motion
contours, and uses a pre-trained AI model to classify objects like people, cars,
and animals in real-time.

⚠️ IMPORTANT: REQUIRED MODEL FILES
----------------------------------
To run this script, you MUST have the following two files in the same folder:
1. deploy.prototxt (The Model Architecture)
2. mobilenet_iter_73000.caffemodel (The Pre-trained Weights)

Works with: Live Webcams (0) and Recorded Video Files (.mp4, .avi, etc.)
"""

# ==========================================
# Step 1: Import Required Libraries
# ==========================================
import cv2
import numpy as np
import time

# ==========================================
# Step 2: Load the AI Model & Classes
# ==========================================
# deploy.prototxt: The architecture/structure of the neural network.
# mobilenet_iter_73000.caffemodel: The pre-trained weights (learned intelligence).
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of object categories the model can recognize
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# ==========================================
# Step 3: Video Setup & Background Subtraction
# ==========================================
# Set source: Use 0 for Webcam or "path/to/video.mp4" for a file
video_source = "test_video.mp4"
cap = cv2.VideoCapture(video_source)

# history: Number of frames to remember (higher means objects stay 'new' longer).
# varThreshold: Sensitivity to noise (higher means ignore small changes like wind/leaves).
# detectShadows: If True, identifies shadows without marking them as real objects.
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# ==========================================
# Step 4: Main Processing Loop
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Adjust playback speed (Optional: 0.05 makes video look more natural)
    time.sleep(0.05)

    # Apply Background Subtraction to create a motion mask (black and white)
    fgmask = fgbg.apply(frame)

    # ==========================================
    # Step 5: Image Cleaning (Dilation)
    # ==========================================
    # Create a 5x5 brush (kernel) to fill holes in the motion mask
    kernel = np.ones((5, 5), np.uint8)

    # Dilation thickens the white areas to bridge gaps in moving objects
    # iterations=2 means we apply the thickening twice to remove noise
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # ==========================================
    # Step 6: Finding Object Contours
    # ==========================================
    # Find the external boundaries (outlines) of the white moving parts
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter: Skip small movements (noise) below 2000 pixels in size
        if cv2.contourArea(cnt) < 2000:
            continue

        # Get the bounding box coordinates (x, y, width, height) for the motion
        (x, y, w, h) = cv2.boundingRect(cnt)

        # ==========================================
        # Step 7: AI Prediction (The Blob)
        # ==========================================
        # Convert the frame into a 'Blob' (Special format AI understands)
        # 1. Resize to 300x300 (Model requirement)
        # 2. Scale pixels (0.007843) to normalize data
        # 3. Subtract mean (127.5) to center the color data
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Pass the blob through the neural network
        net.setInput(blob)
        detections = net.forward()

        # ==========================================
        # Step 8: Drawing Results (Labels & Boxes)
        # ==========================================
        # Loop through all objects detected by the AI
        for i in range(detections.shape[2]):
            # Get the confidence score (how sure the AI is)
            confidence = detections[0, 0, i, 2]

            # Only show results with more than 50% confidence
            if confidence > 0.5:
                # Extract the Object Index (ID) and map it to our CLASSES list
                idx = int(detections[0, 0, i, 1])

                # Scale the AI coordinates (0 to 1) back to actual frame pixel size
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Format the text label (e.g., "person: 98.5%")
                label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"

                # Draw the bounding box around the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Draw the text label slightly above the box
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the final output with detection boxes
    cv2.imshow('AI Motion Detection', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()