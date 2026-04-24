# ==============================================================================
# PROJECT: Video Classification using Vision Transformer (ViT) from Scratch
# FILENAME: 04_video_transformer_scratch.py
# DESCRIPTION: Building a Transformer model to classify videos (e.g., Running vs Sitting)
#              by converting video frames into patches and using Self-Attention.
# ==============================================================================

# STEP 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# STEP 2: Set Model Parameters
# These define how the model "sees" the video
num_frames = 8  # Number of frames the model looks at to understand motion
image_size = 32  # Resizing frames to 32x32 for faster processing
channels = 3  # RGB Colors
patch_size = 4  # Dividng 32x32 image into small 4x4 squares (8x8 = 64 total patches)
projection_dim = 64  # Each 4x4 patch is converted into a vector of 64 numbers

# STEP 3: Build the Input Layer
# Shape: (Batch, Frames, Height, Width, Channels)
inputs = layers.Input(shape=(num_frames, image_size, image_size, channels))

# STEP 4: Patching & Projection (Conv3D)
# We use Conv3D to extract both Spatial (image) and Temporal (time/motion) data.
# It takes 2 frames at a time and cuts them into 4x4 patches.
patches = layers.Conv3D(
    filters=projection_dim,
    kernel_size=(2, patch_size, patch_size),
    strides=(2, patch_size, patch_size),
    padding="valid"
)(inputs)

# STEP 5: Reshape for Transformer
# Transformers don't understand 5D video data; they need a 1D sequence (like words in a sentence).
# We flatten the 256 patches into one long list for the model to read.
num_patches = patches.shape[1] * patches.shape[2] * patches.shape[3]
x = layers.Reshape((num_patches, projection_dim))(patches)

# STEP 6: Multi-Head Self-Attention
# This is the "Brain." It calculates the relationship between different parts of the video.
# It asks: "Does the hand moving in Frame 1 relate to the body moving in Frame 8?"
attention_output = layers.MultiHeadAttention(
    num_heads=4, key_dim=projection_dim
)(x, x)

# STEP 7: Add & Normalize (Residual Connection)
# We add the original data back to the attention output to prevent data loss.
# LayerNormalization keeps the numbers stable (between 0 and 1).
x = layers.Add()([attention_output, x])
x = layers.LayerNormalization()(x)

# STEP 8: MLP (Feed Forward Network)
# This part processes the features learned. We expand to 128 units, then back to 64.
y = layers.Dense(projection_dim * 2, activation="gelu")(x)
y = layers.Dropout(0.3)(y)  # Turn off 30% of neurons to prevent "overfitting" (memorization)
y = layers.Dense(projection_dim)(y)

# Final connection and normalization
x = layers.Add()([y, x])
x = layers.LayerNormalization()(x)

# STEP 9: Final Output (Classification)
# GlobalAveragePooling summarizes all 256 patches into one final decision vector.
representation = layers.GlobalAveragePooling1D()(x)
representation = layers.Dropout(0.5)(representation)
# Output layer with 2 classes (Running/Sitting) using Softmax for probabilities.
outputs = layers.Dense(2, activation="softmax")(representation)

# STEP 10: Compile the Model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Low learning rate (0.0001) helps the model learn carefully without "jumping" over the solution.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# STEP 11: Data Loading (Smart Sampling)
all_videos = []
all_labels = []
classes = ['Running', 'Sitting']

for label_index, class_name in enumerate(classes):
    folder_path = f'data/{class_name}'
    if not os.path.exists(folder_path): continue

    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []

        # We divide the video into 8 equal parts (Start, Middle, End)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_interval = max(int(total_frames / num_frames), 1)

        for i in range(num_frames):
            # Jump to the specific frame using the slider logic (cap.set)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval)
            ret, frame = cap.read()
            if not ret: break

            # Resize and Normalize the frame (0 to 1 range)
            frame = cv2.resize(frame, (image_size, image_size))
            frame = frame / 255.0
            frames.append(frame)
        cap.release()

        if len(frames) == num_frames:
            all_videos.append(frames)
            all_labels.append(label_index)

# STEP 12: Prepare for Training
X_train = np.array(all_videos)
y_train = np.array(all_labels)

# STEP 13: Train the Model
if len(X_train) > 0:
    print(f"Data Prepared: {X_train.shape}")
    print("Starting Training...")
    # Small batch size (2) to save RAM and prevent crashes.
    model.fit(X_train, y_train, epochs=3, batch_size=2)
else:
    print("Error: No video data found in data/ folder!")