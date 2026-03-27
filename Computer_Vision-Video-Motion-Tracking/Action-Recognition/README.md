# 🎥 Action Recognition: Architectures & Training Pipelines

Welcome to the **Action Recognition** repository. This project is a comprehensive guide designed to demonstrate how Deep Learning models "see" and "understand" human actions in video data. 

The repository is divided into two main sections: **Architectures** (The Brains) and **Training Pipelines** (The Engines).

---

## 📂 Repository Structure

### 🧠 01. Architectures & Brain Structures
This folder focuses on the mathematical structure and logic of different Neural Networks. These scripts define the "Brain" of the AI without the training overhead.

*   **`01_Action_Recognition_CNN_LSTM.py`**: Combines spatial feature extraction (CNN) with sequential memory (LSTM).
*   **`02_Action_Recognition_3D_CNN.py`**: Uses 3D filters to extract features across space and time simultaneously.
*   **`03_Action_Recognition_Two_Stream_Networks.py`**: A dual-input logic that looks at static images and motion (Optical Flow) separately.
*   **`04_Action_Recognition_ConvLSTM.py`**: A hybrid approach where convolutional layers are embedded inside LSTM cells for better spatial-temporal fusion.

---

### ⚙️ 02. Training Pipelines & Engines
This folder focuses on the **Step-by-Step workflow** of training a model. These are lightweight "practice engines" designed to show you how to process data and feed it into a model using small datasets (e.g., Running vs. Sitting).

*   **`05_Unified_Action_Recognition_Models.py`**: A streamlined pipeline for training basic action classifiers.
*   **`06_Two_Stream_Network_Trainer.py`**: A detailed guide on preparing **Spatial data** (RGB) and **Temporal data** (Optical Flow), and how to train a multi-input model.

---

## 🚀 Key Learning Concepts

In this repository, you will explore:
1.  **Data Normalization:** Why we divide pixel values by 255 to help the AI learn faster and maintain stability.
2.  **Optical Flow:** Using OpenCV to track pixel movement between video frames to capture motion.
3.  **Multi-Input Logic:** How to merge different types of data (Appearance + Motion) into a single decision-making model.
4.  **Spatial vs. Temporal:** Understanding the difference between "What an object looks like" and "How an object moves."

---

## 🛠️ Prerequisites

To run these scripts, you will need the following libraries installed:

```bash
pip install tensorflow opencv-python numpy

## 📝 Note for Users

These scripts are designed for educational purposes. The training engines use small datasets (approx. 10 videos) to focus on the Logic, Pipeline, and Steps of the code. They are perfect for learning the architecture and flow before scaling up to massive datasets like UCF101 or HMDB51.

Maintained by: Abdiftah Ahmed Bashiir
Focus: Computer Vision | Deep Learning | Action Recognition