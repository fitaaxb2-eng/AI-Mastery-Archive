"""
FILE NAME: 03_Action_Recognition_Two_Stream_Networks.py
CORE FOCUS: DUAL-PATH ARCHITECTURE (SPATIAL & TEMPORAL)
PURPOSE: Combining static appearance (RGB) with motion patterns (Optical Flow).
DESCRIPTION: This script builds a model that looks at "What is in the scene" and "How it is moving" separately, then merges them.
"""

# ==========================================
# STEP 1: IMPORTING THE BUILDING BLOCKS
# ==========================================
# Loading TensorFlow and Keras tools to build the dual-stream neural network.
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# STEP 2: SETTING THE MODEL RULES (HYPERPARAMETERS)
# ==========================================
# Defining the size of the input data and the number of categories to recognize.
IMG_HEIGHT = 64       # Height of the video frame in pixels
IMG_WIDTH = 64        # Width of the video frame in pixels
CHANNELS_RGB = 3      # Standard color images (Red, Green, Blue)
CHANNELS_FLOW = 20    # Motion data (10 frames of movement x 2 directions [u, v])
NUM_CLASSES = 5       # The total number of actions to identify (e.g., Walking, Running)

# ==========================================
# STEP 3: BUILDING THE SPATIAL STREAM (The "Appearance" Path)
# ==========================================
# This part of the brain looks at a single frame to identify objects (e.g., a ball, a person).
input_spatial = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS_RGB), name="Spatial_Input")

s = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_spatial) # Detecting basic shapes
s = layers.MaxPooling2D((2, 2))(s)                                              # Shrinking data to save memory
s = layers.BatchNormalization()(s)                                              # Keeping the learning stable

s = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(s)             # Detecting complex objects
s = layers.MaxPooling2D((2, 2))(s)                                              # Further shrinking
s = layers.Flatten()(s)                                                         # Changing 2D image data into a long list
s = layers.Dense(256, activation='relu')(s)                                     # Extracting high-level visual features

# ==========================================
# STEP 4: BUILDING THE TEMPORAL STREAM (The "Motion" Path)
# ==========================================
# This part looks at "Optical Flow" to understand the direction and speed of movement.
input_temporal = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS_FLOW), name="Temporal_Input")

t = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_temporal) # Detecting initial movement
t = layers.MaxPooling2D((2, 2))(t)
t = layers.BatchNormalization()(t)

t = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(t)              # Detecting action patterns
t = layers.MaxPooling2D((2, 2))(t)
t = layers.Flatten()(t)
t = layers.Dense(256, activation='relu')(t)                                      # Extracting high-level motion features

# ==========================================
# STEP 5: THE FUSION (Merging the Two Paths)
# ==========================================
# Combining the "Appearance" and "Motion" knowledge into one single brain.
merged = layers.Concatenate()([s, t])

# The final decision-making layers
x = layers.Dense(128, activation='relu')(merged)      # Processing the combined information
x = layers.Dropout(0.5)(x)                            # Preventing the model from just memorizing (overfitting)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x) # Final guess (Probability for each action)

# ==========================================
# STEP 6: ASSEMBLING AND COMPILING
# ==========================================
# Finalizing the model and choosing how it learns.
model = models.Model(inputs=[input_spatial, input_temporal], outputs=output)

# Instructing the model on how to improve during training
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Displaying the full architecture of the brain
model.summary()

# -------------------------------------------------------------------------
# QUICK EXPLANATION:
# 1. Spatial Stream: Looks at 1 RGB frame. It answers: "What objects are there?"
# 2. Temporal Stream: Looks at 10 Flow frames. It answers: "How are things moving?"
# 3. Fusion: It joins both answers to make a final expert prediction.
# -------------------------------------------------------------------------
