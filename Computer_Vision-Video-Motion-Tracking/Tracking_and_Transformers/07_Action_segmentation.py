# ==============================================================================
# PROJECT: Action Segmentation (Video Timeline Analysis)
# FILENAME: 07_Action_segmentation.py
# DESCRIPTION: This script monitors a video and classifies the action taking place
#              based on the frame number (Timeline Logic).
# ==============================================================================

# STEP 1: Import necessary libraries
import cv2            # Indhaha computer-ka: furida iyo akhrinta video-ga.
import numpy as np    # Xisaabaadka arrays-ka ee sawirada.
import os             # Maamulka system-ka iyo folder-ka.
import tensorflow as tf # Maskaxda AI-da (Neural Networks).
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, Flatten, LSTM, Dense, Input

# STEP 2: Configure System Environment
# Waa inaan daminaa xisaabaadka xawaaraha Intel (oneDNN) si looga fogaado fariimaha "Warnings" ee aan loo baahnayn.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Waxaan dhigaynaa heerka fariimaha '2' si aynu u qarino digniinta aan muhiimka ahayn ee TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# STEP 3: Define the Action Segmentation Model (The Brain)
model = Sequential([
    Input(shape=(None, 64, 64, 3)),
    TimeDistributed(Conv2D(16, (3, 3), activation='relu')), # Soo saarista sifooyinka (Features)
    TimeDistributed(Flatten()),                             # U bedelida xariiq hal-beeg ah
    LSTM(32, return_sequences=True),                        # Xasuusta (Time-series memory)
    Dense(3, activation='softmax')                          # Go'aanka ugu dambeeya (3-da Action)
])

# STEP 4: Define Action Logic (Timeline Rules)
def get_action_label(f_id):
    """
    Function-kani waa 'Garsooraha'. Wuxuu eegayaa lambarka frame-ka
    wuxuuna go'aaminayaa action-ka uu qofku ku jiro.
    """
    if 0 <= f_id <= 94:
        return "1. Walaaqid (Stirring)"
    elif 95 <= f_id <= 199:
        return "2. Biyo miirid & saxan saarid"
    elif 200 <= f_id <= 340:
        return "3. Mar kale: Biyo miirid & saxan saarid"
    else:
        return "Dhamaadka Video-ga"

# STEP 5: Setup Video Source and Window
video_path = "../cooking.mp4"
cap = cv2.VideoCapture(video_path)

window_name = 'Day 26: Action Segmentation - Lesson'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Daaqad la waynayn karo
cv2.resizeWindow(window_name, 700, 500)         # Cabirka daaqadda

# STEP 6: Process Video Frames
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1 # Kordhinta saacadda Timeline-ka

    # Helitaanka Action-ka hadda socda
    current_action = get_action_label(frame_id)

    # Annotations: Qoraalka dusha ka saar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'FRAME: {frame_id}', (40, 70), font, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f'ACTION: {current_action}', (40, frame.shape[0] - 50), font, 1.0, (0, 255, 255), 3)

    # Show result
    cv2.imshow(window_name, frame)

    # STEP 7: Keyboard Interaction
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'): # Ka bixida barnaamijka
        break
    elif key == ord('p'): # Hakin (Pause)
        print(f"Video-gu wuxuu halkan ku hakaday: Frame {frame_id}")
        cv2.waitKey(-1)

# STEP 8: Cleanup Resources
cap.release()
cv2.destroyAllWindows()