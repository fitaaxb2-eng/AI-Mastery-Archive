"""
================================================================================
project: Action Recognition (CNN + LSTM)
TITLE: Building a Hybrid "LRCN" Architecture for Video Classification
GOAL: To extract spatial features using CNN and temporal sequences using LSTM.
      - CNN (MobileNetV2): Acts as the "Eyes" to see individual frames.
      - LSTM: Acts as the "Memory" to understand movement over time.
================================================================================
"""

# ------------------------------------------------------------------------------
# STEP 1: Importing Necessary Libraries
# We are importing TensorFlow and Keras components. These are the tools 
# we need to build, layers, and compile our Deep Learning model.
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, LSTM, Dense, 
                                     Flatten, TimeDistributed, Dropout)
from tensorflow.keras.applications import MobileNetV2

# ------------------------------------------------------------------------------
# STEP 2: Defining Hyperparameters (Input Shapes)
# Here we define the dimensions of our video. Video data is 5D: 
# (Batch_Size, Sequence_Length, Height, Width, Channels).
# ------------------------------------------------------------------------------
NUM_FRAMES = 20    # We take 20 frames per video to recognize the action
IMG_HEIGHT = 224   # Standard height for MobileNetV2 input
IMG_WIDTH = 224    # Standard width for MobileNetV2 input
CHANNELS = 3       # RGB Color channels (Red, Green, Blue)
NUM_CLASSES = 5    # The number of actions we want the model to learn

# ------------------------------------------------------------------------------
# STEP 3: Initializing the CNN Backbone (Feature Extractor)
# We use MobileNetV2, which is a pre-trained model. It already knows how to 
# recognize shapes and objects from millions of images (ImageNet).
# ------------------------------------------------------------------------------
# We set include_top=False to remove the final classification layer.
video_cnn = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

# Freeze the weights so we don't change the CNN's existing knowledge.
video_cnn.trainable = False 

# ------------------------------------------------------------------------------
# STEP 4: Building the Main Sequential Model
# We use 'TimeDistributed' to wrap our CNN. This allows the model to process 
# each of the 20 frames through the same CNN weights sequentially.
# ------------------------------------------------------------------------------
model = Sequential(name="Action_Recognition_Model")

# Apply CNN to every frame in the sequence
model.add(TimeDistributed(video_cnn, input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS)))

# Flatten the CNN output so it can be fed into the LSTM
model.add(TimeDistributed(Flatten()))

# ------------------------------------------------------------------------------
# STEP 5: Temporal Learning with LSTM (The Memory)
# The LSTM layer looks at how the features change from frame 1 to frame 20. 
# This is how the model understands "Action" or "Motion".
# ------------------------------------------------------------------------------
model.add(LSTM(64, return_sequences=False)) # 64 units of long-term memory
model.add(Dropout(0.5))                      # Prevents overfitting (memorization)

# ------------------------------------------------------------------------------
# STEP 6: Classification and Decision Making
# These final 'Dense' layers act as the brain's decision-making center 
# to finalize which action category the video belongs to.
# ------------------------------------------------------------------------------
model.add(Dense(128, activation='relu'))      # Extra learning layer for complex patterns
model.add(Dense(NUM_CLASSES, activation='softmax')) # Final probability for each class

# ------------------------------------------------------------------------------
# STEP 7: Model Compilation
# We tell the model to use the 'Adam' optimizer to learn and 
# 'Categorical Crossentropy' to calculate the error during training.
# ------------------------------------------------------------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the complete structure of our model
model.summary()

"""
FINAL NOTES:
1. Spatial Features: Handled by CNN (Eyes).
2. Temporal Features: Handled by LSTM (Memory).
"""
