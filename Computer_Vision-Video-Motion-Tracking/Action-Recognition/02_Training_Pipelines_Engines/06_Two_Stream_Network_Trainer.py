"""
FILE NAME: 06_Two_Stream_Network_Trainer.py
CORE FOCUS: PRACTICE MODEL - VIDEO CLASSIFICATION (Two-Stream CNN)
PURPOSE: Demonstrating the training workflow for 'Running' vs 'Sitting'.

NOTE: This is a conceptual script designed to show the structure of 
Two-Stream Networks. It uses a tiny dataset (10 videos) to focus on 
the coding logic rather than model accuracy.

================================================================================
QUICK START GUIDE (PREREQUISITES):
---------------------------------------------------------
1. FOLDER SETUP:
   Create a folder named 'data' in the same directory as this script.
   Inside 'data/', create two subfolders: 'Running/' and 'Sitting/'.

2. DATA COLLECTION:
   - Put 5 short clips in 'data/Running/'
   - Put 5 short clips in 'data/Sitting/'

3. WHY TWO-STREAM?
   - Spatial Stream: Learns the "Object" (Appearance).
   - Temporal Stream: Learns the "Action" (Motion/Optical Flow).
================================================================================
"""

# ==========================================
# STEP 1: IMPORTING THE BUILDING BLOCKS
# ==========================================
import cv2  # Library for video and image processing
import numpy as np  # Used for handling numerical arrays and matrices
import os  # Used for navigating folders and file paths
import tensorflow as tf  # The main deep learning framework
from tensorflow.keras import layers, models  # Tools to build neural network layers

# ==========================================
# STEP 2: DATA PREPARATION (SPATIAL & TEMPORAL)
# ==========================================
# In this stage, we convert raw videos into two types of mathematical data:
# 1. Spatial: Static RGB frames (224x224x3).
# 2. Temporal: Motion vectors using Optical Flow (224x224x2).

x_spatial = []  # To store RGB frames
x_temporal = []  # To store Motion (Optical Flow) frames
y_labels = []
classes = ['Running', 'Sitting']

print("--- [PROCESS] Extracting Appearance and Motion Data ---")

for label_index, class_name in enumerate(classes):
    folder_path = f'data/{class_name}'

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found!")
        continue

    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)

        # 1. Capture Spatial Frame (The appearance of the scene)
        ret, frame1 = cap.read()
        if not ret:
            cap.release()
            continue

        # Normalization (0 to 1): 
        # Computers process values between 0 and 1 much faster than 0-255.
        s_frame = cv2.resize(frame1, (224, 224)) / 255.0

        # 2. Capture Temporal Frame (Calculating Motion between two frames)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        ret, frame2 = cap.read()
        if not ret:
            cap.release()
            continue
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate Farneback Optical Flow (Movement data)
        # This creates a 2-channel array representing (x, y) pixel movement.
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t_frame = cv2.resize(flow, (224, 224)) / 255.0

        x_spatial.append(s_frame)
        x_temporal.append(t_frame)
        y_labels.append(label_index)
        cap.release()

# Convert data into Numpy arrays for TensorFlow compatibility
X_S = np.array(x_spatial)
X_T = np.array(x_temporal)
Y = np.array(y_labels)

print(f"Done! Prepared {len(X_S)} training samples.")

# ==========================================
# STEP 3: BUILDING THE TWO-STREAM ARCHITECTURE
# ==========================================
# We build a multi-input model that processes images and motion separately 
# before merging them for the final classification.

# --- Stream 1: Spatial Path (Looks at the frame) ---
input_s = layers.Input(shape=(224, 224, 3), name="Spatial_Input")
s = layers.Conv2D(32, (3, 3), activation='relu')(input_s)
s = layers.MaxPooling2D((2, 2))(s)
s = layers.GlobalAveragePooling2D()(s)

# --- Stream 2: Temporal Path (Looks at the motion) ---
input_t = layers.Input(shape=(224, 224, 2), name="Temporal_Input")
t = layers.Conv2D(32, (3, 3), activation='relu')(input_t)
t = layers.MaxPooling2D((2, 2))(t)
t = layers.GlobalAveragePooling2D()(t)

# --- Merge Layer: Combining both brains into one decision ---
combined = layers.Concatenate()([s, t])
dense = layers.Dense(64, activation='relu')(combined)
output = layers.Dense(2, activation='softmax')(dense)

# Define the final Multi-Input Model
model = models.Model(inputs=[input_s, input_t], outputs=output)

# ==========================================
# STEP 4: COMPILING AND TRAINING
# ==========================================
# This is the final step where the model starts learning from the patterns 
# we extracted in Step 2.

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- [START] Training the Two-Stream Network ---")

# We use a small batch_size and 10 epochs to observe the training logic.
model.fit([X_S, X_T], Y, epochs=10, batch_size=2)

print("\n--- [FINISH] Training completed successfully! ---")
print("Focus of this script: Understanding the architecture & training workflow.")