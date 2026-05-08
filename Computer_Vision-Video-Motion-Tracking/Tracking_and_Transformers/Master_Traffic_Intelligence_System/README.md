# 🚦 VelocityGuard AI: Intelligent Traffic Monitor

VelocityGuard AI is a sophisticated real-time traffic surveillance and speed enforcement system. Built with **Python**, **YOLOv8**, and **OpenCV**, this project detects vehicles, tracks their movement, calculates real-time speed, and automatically logs snapshots of speed violators.

---

## ✨ Key Features

*   **Real-Time Detection:** Identifies cars, trucks, and buses using the YOLOv8 neural network.
*   **Object Tracking:** Assigns unique IDs to every vehicle to monitor movement across frames.
*   **Speed Estimation:** Calculates speed based on pixel displacement and custom scaling factors.
*   **Automatic Violation Logging:** High-definition snapshots are captured and saved when a vehicle exceeds the speed limit.
*   **Live Analytics Dashboard:** A dedicated side-panel showing real-time stats for the most recent vehicles.
*   **HD Coordinate Mapping:** Intelligent scaling that maps AI detections from low-resolution processing frames back to original HD video for high-quality evidence.

---

## 🛠️ Project Architecture

The project is divided into three main modules for better organization:

1.  **`main.py`**: The central controller that manages the video stream, detection loop, and speed logic.
2.  **`tracker.py`**: The mathematical engine that tracks vehicle IDs using Euclidean distance.
3.  **`utils.py`**: The UI and record-keeping module that draws the dashboard and saves violation images.

---

## 🚀 Getting Started

### Step 1: Install Dependencies
Ensure you have Python installed, then run the following command to install the required libraries:

```bash
pip install opencv-python ultralytics numpy
Step 2: Prepare Your Video
Place your traffic video in the project root folder and ensure it is named traffic_video.mp4 (or update the filename in main.py and choose any vedio you want).
Step 3: Run the System
Execute the main script to start monitoring:
code
Bash
python main.py
📂 File Structure
code
Text
VelocityGuard_AI/
├── main.py              # Main Execution Logic
├── tracker.py           # Object Tracking Engine
├── utils.py             # UI Design & Snapshot Utility
├── violations/          # Auto-generated folder for snapshots
└── traffic_video.mp4   # Your Input Video
⚙️ How It Works (Step-by-Step)
Preprocessing: The system resizes the video for faster AI processing while keeping the original HD frame in memory.
Detection: YOLOv8 identifies vehicles in the frame.
Tracking: The Tracker class calculates the center of each vehicle and matches it to existing IDs based on distance.
Speed Logic: By comparing the vehicle's position between current and previous frames, the system calculates the velocity.
Evidence Collection: If the speed exceeds 40 km/h, the system triggers a snapshot. It translates the coordinates from the small frame back to the HD frame to save a crystal-clear image of the violator.
