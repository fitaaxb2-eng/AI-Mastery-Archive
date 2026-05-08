import cv2
from ultralytics import YOLO
from tracker import Tracker
from utils import draw_dashboard, save_snapshot

# Step 1: System Setup
model = YOLO('yolov8n.pt')  # Load AI Model
tracker = Tracker()  # Initialize Tracker
cap = cv2.VideoCapture("traffic_video.mp4")  # Load Video Source or try any vedios traffic you want

vehicle_speed = {}  # Stores current speed of cars
vehicle_prev_pos = {}  # Stores previous Y-position for speed calculation
violation_saved = set()  # Tracks which IDs have already been snapped

while True:
    # Step 2: Read Frame and Resize for AI Processing
    ret, original_frame = cap.read()
    if not ret: break

    orig_h, orig_w, _ = original_frame.shape
    new_w, new_h = 1000, 600
    frame = cv2.resize(original_frame, (new_w, new_h))

    # Calculate scale factor to map small frame back to original HD frame
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    # Step 3: AI Detection (Vehicle Search)
    # Using stream=True for better memory performance
    results = model(frame, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
            cls = int(box.cls[0])  # Get class ID

            # Only track cars, trucks, and buses
            if model.names[cls] in ['car', 'truck', 'bus']:
                detections.append([x1, y1, x2 - x1, y2 - y1])

    # Step 4: Update Tracker with current detections
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x, y, w, h, vid = obj
        cx, cy = (x + x + w) // 2, (y + y + h) // 2

        # Step 5: Speed Calculation Logic
        if vid in vehicle_prev_pos:
            # Distance moved in pixels
            dist = abs(cy - vehicle_prev_pos[vid])
            # Convert pixels to speed (Adjust 1.5 multiplier based on your camera)
            speed = int(dist * 1.5)
            vehicle_speed[vid] = speed

            # Step 6: Handling Speed Violations
            if speed > 40:
                color = (0, 0, 255)  # Red for danger
                label = "ALERT: SPEEDING!"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # Capture snapshot only once per vehicle ID
                if vid not in violation_saved:
                    # Convert coordinates back to HD scale for high-quality snapshot
                    x_hd, y_hd = int(x * scale_x), int(y * scale_y)
                    w_hd, h_hd = int(w * scale_x), int(h * scale_y)
                    save_snapshot(original_frame, x_hd, y_hd, w_hd, h_hd, vid, speed)
                    violation_saved.add(vid)
            else:
                color = (0, 255, 0)  # Green for safe
                label = f"ID:{vid} {speed}km/h"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update memory for next frame
        vehicle_prev_pos[vid] = cy

    # Step 7: Display the final result with Dashboard
    final_view = draw_dashboard(frame, vehicle_speed)
    cv2.imshow("VelocityGuard AI - Live Monitoring", final_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()