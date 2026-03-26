"""
FILE NAME: 05_Unified_Action_Recognition_Models.py
CORE FOCUS: PRACTICE MODEL - VIDEO CLASSIFICATION
PURPOSE: Training a simple model to recognize 'Running' vs 'Sitting'.

NOTE: This is a lightweight script designed for practice and learning.
It is not intended for high-accuracy production use.

================================================================================
QUICK START GUIDE (PREREQUISITES):
---------------------------------------------------------
1. FOLDER SETUP:
   Create a folder named 'data' in the same directory as this script.
   Inside 'data/', create two subfolders:

   [Your Project Folder]
    └── 05_Unified_Action_Recognition_Models.py
    └── data/
         ├── Running/   <-- (Put 5 videos of running here)
         └── Sitting/   <-- (Put 5 videos of sitting here)

2. DATA COLLECTION (PRACTICE ONLY):
   You only need a small dataset to see how it works.
   - Download 5 short videos for 'Running'.
   - Download 5 short videos for 'Sitting'.

3. WHERE TO FIND VIDEOS (FREE WEBSITES):
   - Pixabay: https://pixabay.com/videos/
   - Pexels:  https://www.pexels.com/search/videos/
   (Search for "Running" and "Sitting" and download short clips).

4. IMPORTANT:
   Ensure this script and the 'data' folder are in the same location
   on your computer so the code can find the videos.
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
# STEP 2: DATA PREPROCESSING (READING VIDEOS)
# ==========================================
# Initializing empty lists to store video data and labels
all_videos = []
all_labels = []
# Defining the action categories (Make sure these match your folder names)
classes = ['Running', 'Sitting']

# IMPORTANT: Adjust image size based on the model you choose!
# CNN+LSTM usually uses 224, while 3D-CNN/ConvLSTM might use 64 for speed.
IMG_SIZE = 224

# Loop through each class folder to read videos
for label_index, class_name in enumerate(classes):
    folder_path = f'data/{class_name}'  # Path to the specific action folder

    # Iterate through every video file in the folder
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)

        cap = cv2.VideoCapture(video_path)  # Open the video file
        frames = []  # List to store individual frames of a video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Count total frames in video

        # Calculate how many frames to skip to get exactly 20 evenly spaced frames
        skip_interval = max(int(total_frames / 20), 1)

        for i in range(20):
            # Set the position of the next frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval)
            ret, frame = cap.read()  # Read the frame
            if not ret: break  # Stop if the video ends unexpectedly

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Resize frame to standard size
            frame = frame / 255.0  # Normalize pixel values (0 to 1)
            frames.append(frame)  # Add processed frame to the list

        cap.release()  # Close the video file

        # Only add the video if we successfully captured 20 frames
        if len(frames) == 20:
            all_videos.append(frames)
            all_labels.append(label_index)

# Convert lists into NumPy arrays for model compatibility
X = np.array(all_videos)
y = np.array(all_labels)

# ==========================================
# STEP 3: MODEL SELECTION (CHOOSE ONE)
# ==========================================

# --- OPTION 1: CNN + LSTM (Best for complex features) ---
# Using MobileNetV2 as a pre-trained base for feature extraction
cnn_base = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3),
                                             pooling="avg")
cnn_base.trainable = False  # Keep pre-trained weights frozen

model = models.Sequential([
    layers.Input(shape=(20, 224, 224, 3)),  # Input: 20 frames, each 224x224 RGB
    layers.TimeDistributed(cnn_base),  # Apply CNN to every frame individually
    layers.LSTM(128, return_sequences=False),  # Process the sequence of features over time
    layers.Dense(64, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(2, activation='softmax')  # Output layer (2 classes: Running/Sitting)
])

"""
# --- OPTION 2: 3D-CNN (Good for spatial-temporal features) ---
# NOTE: Set IMG_SIZE = 64 before using this.
model = models.Sequential([
    layers.Input(shape=(20, 64, 64, 3)),
    layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
"""

"""
# --- OPTION 3: ConvLSTM (Excellent for motion patterns) ---
# NOTE: Set IMG_SIZE = 64 before using this.
model = models.Sequential([
    layers.Input(shape=(20, 64, 64, 3)),
    layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'),
    layers.BatchNormalization(),
    layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(2, activation='softmax')
])
"""

# ==========================================
# STEP 4: COMPILING AND TRAINING
# ==========================================
# Configure the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting the training process...")
# Train the model with the prepared data
model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

print("Training is complete!")