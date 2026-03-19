\# 🚀 AI Motion Detection \& Object Tracking Suite



This repository contains a collection of Computer Vision scripts designed to detect, track, and classify motion using \*\*OpenCV\*\* and \*\*Deep Learning (MobileNet-SSD)\*\*. It covers everything from basic background subtraction to advanced AI-integrated tracking.



\---



\## 📂 Project Structure



The project is organized into 5 progressive steps, each demonstrating a different computer vision technique:



\### 01. `01\_background\_subtraction.py` 

The foundation of motion detection. This script separates moving foreground objects from the static background using the \*\*MOG2 algorithm\*\*. It identifies what is moving by comparing current frames to a learned background model.



\### 02. `02\_sparse\_optical\_flow.py` 

Uses the \*\*Lucas-Kanade method\*\* to track specific "feature points" (like corners) across video frames. It's efficient and great for tracking the direction of movement for specific objects.



\### 03. `03\_dense\_optical\_flow.py` 

Unlike sparse flow, this tracks \*\*every single pixel\*\* in the frame. It produces a colorful heatmap where different colors represent different directions of motion. Ideal for analyzing full-scene activity.



\### 04. `04\_motion\_detection\_final.py` 

A refined version of motion detection. It uses contours and bounding boxes to highlight moving objects, filters out small camera noise, and provides a clean visual of active targets.



\### 05. `05\_cnn\_and\_motion\_integration.py` 

\*\*The Master Script.\*\* It combines motion detection with a \*\*Convolutional Neural Network (MobileNet-SSD)\*\*. It doesn't just see "motion"—it identifies if the motion belongs to a \*\*person, car, dog, or bicycle\*\*.



\---



\## ✨ Key Features

\*   \*\*Real-time Processing:\*\* Works with live Webcams (`source = 0`) and Recorded Video Files.

\*   \*\*Noise Filtering:\*\* Uses Dilation and Area-thresholding to ignore background distractions like wind or leaves.

\*   \*\*High Accuracy:\*\* Uses Deep Learning to reduce "False Positives" in motion detection.

\*   \*\*Multi-Object Support:\*\* Can detect and label multiple different objects (People, Cars, Bikes) simultaneously.



\---



\## 🧠 AI Model Files (Included)



The following pre-trained model files are included in this folder to make the project "Plug-and-Play":



\*   \*\*`deploy.prototxt`\*\*: This file defines the \*\*Architecture\*\* (the layers) of the MobileNet-SSD neural network.

\*   \*\*`mobilenet\_iter\_73000.caffemodel`\*\*: This contains the \*\*Trained Weights\*\* (the intelligence). It was trained on thousands of images to recognize 21 different classes of objects.



> \*\*Note:\*\* These two files must remain in the same directory as the Python scripts for the AI detection to work.



\---



\## 🛠️ Requirements \& Installation



To run these scripts, you need \*\*Python\*\* installed on your system. You can install the necessary libraries using pip:



```bash

pip install opencv-python numpy

