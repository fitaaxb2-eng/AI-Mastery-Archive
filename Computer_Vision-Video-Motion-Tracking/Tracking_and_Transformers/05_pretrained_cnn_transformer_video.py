# ==============================================================================
# PROJECT: Video Classification using CNN + Transformer (Attention)
# FILENAME: 05_pretrained_cnn_transformer_video.py
# DESCRIPTION: This script uses a pre-trained MobileNetV2 to extract features
#              from video frames and a Multi-Head Attention layer to classify
#              actions (Running vs. Sitting).
# ==============================================================================

# STEP 1: Import necessary libraries
import tensorflow as tf  # Main framework for deep learning
from tensorflow.keras import layers, models, applications  # Components to build the neural network
import numpy as np  # For numerical and array operations
import cv2  # For video processing and frame extraction
import os  # For interacting with the operating system and file paths

# STEP 2: Define Model Parameters
num_frames = 8  # Number of frames to extract from each video
image_size = 224  # Standard width and height for the images
channels = 3  # RGB color channels (Red, Green, Blue)
num_classes = 2  # Number of categories to classify (Running and Sitting)

# STEP 3: Setup Pre-trained Feature Extractor (MobileNetV2)
# We use MobileNetV2 because it is lightweight and efficient for video
base_model = applications.MobileNetV2(
    input_shape=(image_size, image_size, channels), # Define input shape
    include_top=False,  # Remove the final classification layer
    weights='imagenet'  # Use knowledge learned from the ImageNet dataset
)
base_model.trainable = False  # Freeze the weights so they don't change during training

# Create a 'Mini-Model' that converts a single image into a vector of 1280 features
mn_input = layers.Input(shape=(image_size, image_size, channels)) # Define input
mn_out = base_model(mn_input) # Pass input through MobileNet
mn_out = layers.GlobalAveragePooling2D()(mn_out)  # Flatten the features into a single vector
feature_extractor = models.Model(inputs=mn_input, outputs=mn_out) # Finalize feature extractor model

# STEP 4: Build the Full Video Model (TimeDistributed)
# This part handles a sequence of 8 frames at once
inputs = layers.Input(shape=(num_frames, image_size, image_size, channels)) # Sequence input

# TimeDistributed applies the feature extractor to every frame individually
x = layers.TimeDistributed(feature_extractor)(inputs) # Output shape: (Batch, 8, 1280)

# STEP 5: Add Multi-Head Attention (The "Brain")
# This layer allows the model to understand the relationship between the 8 frames
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=1280)(x, x) # Compare frames
x = layers.Add()([x, attention_output])  # Residual connection (add original + attention)
x = layers.LayerNormalization()(x) # Normalize the data for faster learning

# STEP 6: Classification Layers
x = layers.GlobalAveragePooling1D()(x) # Compress the temporal (time) data
x = layers.Dense(128, activation='gelu')(x) # Hidden layer with GELU activation
x = layers.Dropout(0.3)(x) # Prevent overfitting by randomly turning off 30% of neurons
outputs = layers.Dense(num_classes, activation='softmax')(x) # Final output layer (Probability)

# Create and Compile the Model
model = models.Model(inputs, outputs) # Define the final model structure
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Use Adam optimizer
    loss='sparse_categorical_crossentropy', # Loss function for integer labels
    metrics=['accuracy'] # Track accuracy during training
)

model.summary() # Print the architecture summary

# STEP 7: Data Loading and Preprocessing
all_videos = [] # List to store video data
all_labels = [] # List to store labels
classes = ['Running', 'Sitting'] # Define our target classes

for label_index, class_name in enumerate(classes): # Loop through each folder
    folder_path = f'data/{class_name}' # Set folder path
    if not os.path.exists(folder_path): # Skip if folder does not exist
        continue

    for video_name in os.listdir(folder_path): # Loop through videos in folder
        video_path = os.path.join(folder_path, video_name) # Get full video path
        cap = cv2.VideoCapture(video_path) # Open the video file
        frames = [] # List to store frames for this specific video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frame count

        if total_frames == 0: # Skip empty or broken videos
            continue

        # Calculate a skip interval to pick 8 frames evenly from the whole video
        skip_interval = max(int(total_frames / num_frames), 1)

        for i in range(num_frames): # Extract 8 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval) # Jump to specific frame
            ret, frame = cap.read() # Read the frame
            if not ret: break # Stop if reading fails
            frame = cv2.resize(frame, (image_size, image_size)) # Resize to 224x224
            frame = frame / 255.0  # Normalize pixel values (0 to 1)
            frames.append(frame) # Add frame to list
        cap.release() # Release video file

        if len(frames) == num_frames: # Ensure we have exactly 8 frames
            all_videos.append(frames) # Add video sequence to dataset
            all_labels.append(label_index) # Add label to dataset

# Convert lists to NumPy arrays for TensorFlow
X_train = np.array(all_videos)
y_train = np.array(all_labels)

# STEP 8: Start Training
if len(X_train) > 0: # If data was found
    print(f"\n✅ Data Ready for Training: {X_train.shape}")
    model.fit(X_train, y_train, epochs=5, batch_size=2) # Train for 5 epochs
else: # If no data was found
    print("\n❌ No data found! Please check if your folders are correct.")