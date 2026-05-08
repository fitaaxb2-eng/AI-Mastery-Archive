
### . utils.py (UI & Record Management)

import cv2
import numpy as np
import os

# Step 1: Create a folder to store speed violation images if it doesn't exist
if not os.path.exists('violations'):
    os.makedirs('violations')


# Step 2: Define the Dashboard UI
def draw_dashboard(frame, active_vehicles):
    """Creates a dark sidebar and lists the speed of active vehicles."""
    h, w, _ = frame.shape
    sidebar_w = 300

    # Create a black background for the sidebar
    sidebar = np.zeros((h, sidebar_w, 3), dtype=np.uint8)

    # Add Title and Decoration Line
    cv2.putText(sidebar, "VELOCITY GUARD AI", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(sidebar, (20, 70), (280, 70), (255, 255, 255), 1)

    # Step 3: Loop through the last 10 vehicles and display their speed
    y_pos = 120
    for vid, speed in list(active_vehicles.items())[-10:]:
        # Green color for safe speed, Red for over 40km/h
        color = (0, 255, 0) if speed <= 40 else (0, 0, 255)
        text = f"ID: {vid} | Speed: {speed}km/h"
        cv2.putText(sidebar, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 35  # Move down for the next line

    # Combine the video frame and the sidebar side-by-side
    return np.hstack((frame, sidebar))


# Step 4: Define the Snapshot Function for Evidence
def save_snapshot(original_hd_frame, x, y, w, h, vehicle_id, speed):
    """Crops the vehicle from the original HD frame and saves it to disk."""
    h_max, w_max, _ = original_hd_frame.shape

    # Ensure the crop coordinates stay within the image boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_max, x + w), min(h_max, y + h)

    # Crop the image
    crop = original_hd_frame[y1:y2, x1:x2]

    if crop.size > 0:
        filename = f"violations/car_ID_{vehicle_id}_Speed_{speed}kmh.jpg"
        # Save with 100% quality for better evidence zoom
        cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])