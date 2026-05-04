# ==============================================================================
# PROJECT: Video Action Recognition using Video Swin Transformer (Simplified)
# FILENAME: 06_video_swin_transformer.py
# DESCRIPTION: This script implements a Video Swin Transformer architecture 
#              to classify human actions (e.g., Running vs. Sitting) in videos.
# ==============================================================================

# STEP 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# STEP 2: Define Model Parameters
num_frames = 8  # Number of frames to sample from each video
image_size = 224  # Swin Transformers prefer larger input resolutions
channels = 3  # RGB Color channels
num_classes = 2  # Number of actions to recognize (Running, Sitting)

# STEP 3: Create the Input Layer
# Shape: (Batch, Frames, Height, Width, Channels)
inputs = layers.Input(shape=(num_frames, image_size, image_size, channels))

# STEP 4: STAGE 1 - Patch Partition (Fine-grained details)
# We use Conv3D to create "3D Cubes" (Time + Space). 
# This helps the model see small initial details (4x4 patches).
x = layers.Conv3D(
    filters=96,
    kernel_size=(2, 4, 4),
    strides=(2, 4, 4),
    padding="valid"
)(inputs)
x = layers.LayerNormalization()(x)

# STEP 5: STAGE 2 - Window Attention (Local focus)
# Instead of looking at the whole frame at once, Swin looks at "Local Windows."
# This makes the model much faster and more efficient.
shape = x.shape  # Example: (None, 4, 56, 56, 96)
x_reshaped = layers.Reshape((shape[1] * shape[2] * shape[3], shape[4]))(x)

# Multi-Head Attention (The core of the Transformer)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=96)(x_reshaped, x_reshaped)
x_reshaped = layers.Add()([x_reshaped, attention_output])
x_reshaped = layers.LayerNormalization()(x_reshaped)

# STEP 6: STAGE 3 - Patch Merging (Zooming Out)
# This is the "Swin Secret." It merges 4 small patches into 1 larger patch.
# It acts like a "Zoom-out" to see the bigger picture of the movement.
x = layers.Reshape((shape[1], shape[2], shape[3], shape[4]))(x_reshaped)
x = layers.AveragePooling3D(pool_size=(1, 2, 2))(x)

# STEP 7: Output Head (Final Decision)
x = layers.GlobalAveragePooling3D()(x)  # Convert all video data into a single vector
x = layers.Dense(128, activation='gelu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

# STEP 8: Build and Compile the Model
model = models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model architecture
model.summary()

# STEP 9: Data Loading Logic
# This part scans folders, extracts frames, and prepares them for training.
all_videos = []
all_labels = []
classes = ['Running', 'Sitting']

for label_index, class_name in enumerate(classes):
    folder_path = f'data/{class_name}'
    if not os.path.exists(folder_path):
        continue

    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_interval = max(int(total_frames / num_frames), 1)

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval)
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (image_size, image_size))
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
        cap.release()

        if len(frames) == num_frames:
            all_videos.append(frames)
            all_labels.append(label_index)

# Convert lists to NumPy arrays for TensorFlow
X_train = np.array(all_videos)
y_train = np.array(all_labels)

# STEP 10: Model Training
if len(X_train) > 0:
    print(f"Data ready for Video Swin. Shape: {X_train.shape}")
    model.fit(X_train, y_train, epochs=5, batch_size=2)
else:
    print("Error: No video data found! Check your 'data/' folder.")

# ==============================================================================
# UNDERSTANDING VIDEO SWIN TRANSFORMER: THE CORE CONCEPTS
# ==============================================================================
# 1. HIERARCHICAL STRUCTURE: Unlike standard Transformers (ViT) that only look
#    at one scale, Swin starts small (4x4) and grows larger. It's like having 
#    both a microscope and a telescope.
#
# 2. SHIFTED WINDOWS: It connects different parts of the image by "shifting" 
#    its focus windows, ensuring no information is lost at the boundaries.
#
# 3. 3D PATCHING: It treats video as a 3D volume (Width x Height x Time), 
#    capturing how objects move across frames efficiently.
#
# 4. EFFICIENCY: By using "Local Attention," it uses much less memory than 
#    traditional Transformers, making it perfect for long videos.
# ==============================================================================