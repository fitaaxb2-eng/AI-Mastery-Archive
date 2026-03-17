# ======================================================================================
# PROJECT: Dense Optical Flow Tracking (Gunnar-Farneback Algorithm)
# DESCRIPTION: This script detects and visualizes motion in a video stream. 
# It calculates the movement of every single pixel (Dense Flow) and represents:
#   1. Direction of motion -> Using Colors (Hue)
#   2. Speed of motion -> Using Brightness (Value)
# It works with both recorded video files and live webcam feeds.
# ======================================================================================
# Step 1: Importing Libraries
# We use 'cv2' for computer vision tasks and 'numpy' for mathematical operations.
import cv2
import numpy as np

# Step 2: Loading Video Source (Webcam or File)
# - Set 'video_source = 0' to use your LIVE WEBCAM.
# - Set 'video_source = "video.mp4"' to use a recorded file.
video_source = 0  # Changed to 0 for Live Cam as requested
cap = cv2.VideoCapture(video_source)

# Step 3: Preparing the Initial Frame
# We read the first frame to establish a starting point for motion comparison.
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not open video or webcam.")
    exit()

# Step 4: Pre-processing (Resize and Grayscale)
# We resize for faster performance and convert to Grayscale because the
# algorithm only needs light intensity changes, not colors.
frame1 = cv2.resize(frame1, (640, 360))
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Step 5: Creating the HSV Mask
# We create an empty black image (canvas) to draw the motion colors later.
# We set Saturation (index 1) to 255 to make the motion colors bright.
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# Step 6: Processing the Video Loop
# We process each frame one by one as long as the video is running.
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # Step 7: Preparing the Next Frame
    # Resize and convert the current frame to Grayscale for comparison.
    frame2 = cv2.resize(frame2, (640, 360))
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Step 8: Calculating Optical Flow
    # The Farneback algorithm calculates how much each pixel moved between 'prvs' and 'next'.
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Step 9: Converting Motion to Visual Data (Color & Speed)
    # Convert the (X, Y) movement into Polar coordinates (Magnitude and Angle).
    # Magnitude = Speed, Angle = Direction.
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Step 10: Mapping Angle to Color (Hue)
    # Different directions of movement will result in different colors.
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Step 11: Mapping Magnitude to Brightness (Value)
    # Faster moving objects will appear brighter on the screen.
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Step 12: Converting Results to Viewable Image
    # Convert the HSV data back to BGR so we can display it as a standard image.
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 13: Displaying the Output Windows
    cv2.imshow('Original Video', frame2)
    cv2.imshow('Optical Flow (Motion Heatmap)', rgb)

    # Step 14: Exit Strategy
    # Wait 30ms between frames; if user presses 'q', the program stops.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Step 15: Updating the Previous Frame
    # Set the current frame as the 'old' frame for the next loop iteration.
    prvs = next_frame

# Step 16: Releasing Resources
# Close the camera and destroy all open windows to free up computer memory.
cap.release()
cv2.destroyAllWindows()