"""
FILE NAME: 04_Action_Recognition_ConvLSTM.py
CORE FOCUS: SPATIO-TEMPORAL LEARNING (ConvLSTM2D)
PURPOSE: Identifying actions by processing frames and motion simultaneously.
DESCRIPTION: This script uses ConvLSTM2D layers to learn visual features and
             time-based patterns in one single step without needing extra layers.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense

# ==========================================
# STEP 1: DEFINE DATA DIMENSIONS
# ==========================================
# We define the size of our video frames and the number of action categories.
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3        # Color video (RGB)
NUM_CLASSES = 10    # Example: 10 different types of actions

# ==========================================
# STEP 2: BUILD THE ConvLSTM MODEL
# ==========================================
model = Sequential()

# ConvLSTM2D Layer:
# This layer is special because it works like a CNN and LSTM combined.
# 'None' in input_shape means the model is FLEXIBLE; it can accept videos
# with any number of frames (10, 20, or 100 frames).
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                     input_shape=(None, IMG_HEIGHT, IMG_WIDTH, CHANNELS),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())

# Second ConvLSTM layer to refine the features.
# return_sequences=False because we want a single summary of the whole video at the end.
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False))
model.add(BatchNormalization())

# ==========================================
# STEP 3: FLATTEN AND PREDICT
# ==========================================
# Flatten converts the 3D data into a 1D vector so the Dense layer can read it.
model.add(Flatten())

# Fully connected layer to learn complex patterns.
model.add(Dense(256, activation='relu'))

# Output layer: Predicts the final action class using Softmax.
model.add(Dense(NUM_CLASSES, activation='softmax'))

# ==========================================
# STEP 4: MODEL COMPILATION
# ==========================================
# Setting up the optimizer and loss function to start the training process.
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display the architecture of the model
model.summary()

# ==========================================
# WHY USE ConvLSTM? (QUICK SUMMARY)
# ==========================================
# 1. Spatio-Temporal: It learns "What is in the frame" and "How it moves" at the same time.
# 2. 5D Input: It processes data as (Samples, Time, Height, Width, Channels).
# 3. No TimeDistributed: You don't need TimeDistributed layers because ConvLSTM
#    is designed to handle sequences natively.
