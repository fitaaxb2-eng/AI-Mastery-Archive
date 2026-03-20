"""
================================================================================
Project: Video Action Recognition
Module: 01 - CNN + LSTM (LRCN) Architecture
Objective: Building a hybrid model that uses "Eyes" (CNN) to see frames
           and "Memory" (LSTM) to understand the sequence of motion.
================================================================================
"""
# ------------------------------------------------------------------------------
# STEP 1: Importing Essential Libraries
# In this step, we load the "Engine" (TensorFlow) and the specific layers
# needed to build our deep learning architecture.
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LSTM, Dense,
                                     Flatten, TimeDistributed, Dropout)
from tensorflow.keras.applications import MobileNetV2

# ------------------------------------------------------------------------------
# STEP 2: Defining Data Shapes (Hyperparameters)
# We define how many frames (images) are in a video and the size of each image.
# This ensures the model knows the dimensions of the incoming video data.
# ------------------------------------------------------------------------------
NUM_FRAMES = 20    # Number of images per video sequence
IMG_HEIGHT = 224   # Input height for the CNN
IMG_WIDTH = 224    # Input width for the CNN
CHANNELS = 3       # RGB color (Red, Green, Blue)
NUM_CLASSES = 5    # The number of actions we want to classify

# ------------------------------------------------------------------------------
# STEP 3: Initializing the CNN Backbone (The Vision Part)
# We use a pre-trained "MobileNetV2" model. This model has already seen
# millions of images, so it acts as the "Eyes" of our system.
# ------------------------------------------------------------------------------
# include_top=False means we remove the classification layer of MobileNet
video_cnn = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

# We freeze the weights because we don't want to retrain the CNN from scratch.
video_cnn.trainable = False

# ------------------------------------------------------------------------------
# STEP 4: Constructing the LRCN Architecture (CNN + LSTM)
# This is the core structure. We use 'TimeDistributed' to apply the CNN
# to every single frame of the video sequence.
# ------------------------------------------------------------------------------
model = Sequential(name="Action_Recognition_Net")

# Applying the CNN to all 20 frames
model.add(TimeDistributed(video_cnn, input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS)))

# Flattening the features so the LSTM can process them as a list (vector)
model.add(TimeDistributed(Flatten()))

# ------------------------------------------------------------------------------
# STEP 5: Adding the Memory Layer (LSTM)
# The LSTM connects the frames together to understand the "Motion".
# ------------------------------------------------------------------------------
model.add(LSTM(64, return_sequences=False)) # 64 neurons for memory
model.add(Dropout(0.5))                      # Prevents the model from memorizing (overfitting)

# ------------------------------------------------------------------------------
# STEP 6: Final Decision Layers (Output)
# These layers translate the LSTM's memory into a final prediction.
# ------------------------------------------------------------------------------
model.add(Dense(128, activation='relu'))      # Extra learning layer
model.add(Dense(NUM_CLASSES, activation='softmax')) # Outputting probabilities for each action

# ------------------------------------------------------------------------------
# STEP 7: Model Compilation
# We tell the model how to learn (Adam optimizer) and how to measure its
# mistakes (Crossentropy loss).
# ------------------------------------------------------------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the summary of the model layers
model.summary()

"""
QUICK SUMMARY:
1. CNN (MobileNetV2): Extracts visual features from single frames.
2. TimeDistributed: Acts as a bridge to process multiple frames through the CNN.
3. LSTM: Connects the frames over time to understand "Motion" or "Action".
4. Dense: Makes the final decision on what action is being performed.
"""
