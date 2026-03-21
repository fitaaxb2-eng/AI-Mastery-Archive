"""
FILE NAME: 02_Action_Recognition_3D_CNN.py
CORE FOCUS: BRAIN STRUCTURE (ARCHITECTURE)
PURPOSE: Designing the 3D-CNN Spatiotemporal Model Architecture.
DESCRIPTION: This script defines the neural architecture used to process video data.
"""

# ==========================================
# STEP 1: IMPORTING THE BUILDING BLOCKS
# ==========================================
# Loading TensorFlow and Keras components to build the neural layers.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,             # The "Eyes" that see 3D motion and shapes
    MaxPooling3D,       # Tool to simplify and compress 3D data
    Flatten,            # Converts 3D features into a 1D vector
    Dense,              # The "Neurons" for final decision making
    Dropout,            # Prevents the brain from memorizing/overfitting
    BatchNormalization  # Stabilizes and speeds up the learning process
)

# ==========================================
# STEP 2: DEFINE BRAIN INPUT DIMENSIONS
# ==========================================
# Setting the input shape: (Frames, Height, Width, Channels)
NUM_FRAMES = 20    # Number of sequential frames to process
IMG_HEIGHT = 64    # Spatial resolution (Height)
IMG_WIDTH = 64     # Spatial resolution (Width)
CHANNELS = 3       # Color format (RGB)
NUM_CLASSES = 5    # Total action categories to recognize

# ==========================================
# STEP 3: CONSTRUCTING THE BRAIN ARCHITECTURE (3D-CNN)
# ==========================================
# Building the layers that extract features across both space and time.
model = Sequential()

# LAYER 1: Initial Spatiotemporal Feature Extraction
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())

# LAYER 2: Deeper Motion & Pattern Recognition
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())

# LAYER 3: High-level Action Feature Extraction
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())

# LAYER 4: THE REASONING HEAD (Decision Layer)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# ==========================================
# STEP 4: PREPARING THE BRAIN FOR LEARNING (Compile)
# ==========================================
# Setting up the optimization engine and performance metrics.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# STEP 5: VISUALIZING THE ARCHITECTURE SUMMARY
# ==========================================
# Displaying the internal parameters and layer connectivity of the model.
model.summary()

"""
BRAIN LOGIC: WHY 3D-CNN ARCHITECTURE?
-------------------------------------
1. UNIFIED VISION: It processes (X, Y, Time) in one go using 3D kernels.
2. EFFICIENCY: Designed to capture short-term temporal patterns without needing LSTMs.
3. STRUCTURE: A modular design that is easy to scale for more complex actions.
"""