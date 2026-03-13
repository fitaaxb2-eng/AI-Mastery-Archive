# 🌿 PlantVillage AI: 38-Class Plant Disease Classifier & AI Expert

This is a high-performance Artificial Intelligence application designed to help farmers identify plant diseases. It combines **Deep Learning (Computer Vision)** to detect diseases and **Generative AI (LLM)** to provide expert agricultural advice.

---

## 🌟 Key Features
- **38 Disease Classes:** Accurately classifies 38 different types of healthy and diseased plant leaves.
- High Performance: Achieved an impressive 94% validation accuracy during training, ensuring reliable and precise disease detection for farmers.
- **Deep Learning Core:** Built using TensorFlow/Keras with a fine-tuned MobileNetV2 architecture.
- **AI Consultation:** Integrated with **Google Gemini Pro** (named "Khabiirka Dhirta") to chat with farmers and provide solutions.
- **Somali Language Support:** The AI expert communicates primarily in Somali to support local farmers.
- **Interactive Dashboard:** Built with Streamlit for a seamless user experience.

## 📊 Dataset Source
The model was trained on the **PlantVillage Dataset** available on **Kaggle**. It contains over 50,000 images of healthy and infected crop leaves.
- [Link to Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalideeb/plantvillage-dataset)

---

## 🏗️ App Logic (The 15 Steps)
The `app.py` follows a structured 15-step workflow:
1.  **Page Configuration:** Setting up the Streamlit UI (Title & Icon).
2.  **API Setup:** Connecting to Google Gemini via API Key.
3.  **Model Picker:** Automatically selecting the best Gemini model (e.g., Gemini 1.5 Flash).
4.  **Expert Rules:** Defining the AI's persona ("Khabiirka Dhirta") via System Prompts.
5.  **Initialize Gemini:** Activating the LLM with custom instructions.
6.  **Load Image Model:** Loading the `plantvillage_model.keras` file.
7.  **Load Class Names:** Mapping model indices to disease names using `class_names.json`.
8.  **Chat Memory:** Initializing session state to remember the conversation.
9.  **Sidebar Settings:** Options to reset chat or view AI model info.
10. **File Uploader:** Interface for users to upload leaf images (JPG/PNG).
11. **Image Processing:** Resizing and prepping images for the Deep Learning model.
12. **Prediction:** Running the Keras model to identify the disease.
13. **Show Results:** Displaying the diagnosis and confidence level.
14. **Auto AI Summary:** Gemini automatically generates a summary and 3 tips for the farmer.
15. **Chat Interface:** An interactive chat window to ask the AI further questions.

---

# 🌿 PlantVillage AI: Aqoonsiga Cudurada Dhirta (38 Nooc)

Kani waa codsi horumarsan oo adeegsada Garaadka Macmalka ah (AI) si looga caawiyo beeralayda aqoonsiga cudurada dhirta. 
Waxuxuu isku darayaa **Deep Learning** oo sawirada baara iyo **Generative AI** oo bixiya talooyin beereed.
 **Saxnaan Sare (Accuracy):** Model-ku wuxuu ku shaqaynayaa ilaa **94% saxnaan ah**, taas oo xaqiijinaysa in natiijadu tahay mid lagu kalsoonaan karo.
- **TensorFlow/Keras:** Lagu dhisay model-ka barashada mashiinka.

## 🛠️ Qalabka la isticmaalay (Tech Stack)
- **TensorFlow/Keras:** Lagu dhisay model-ka barashada mashiinka.
- **Streamlit:** Lagu dhisay qaabka uu u muuqdo app-ka (Frontend).
- **Google Gemini Pro:** Khabiirka AI ee kula talinaya.
- **Python:** Luuqadda rasmiga ah ee mashruuca.

## 📂 Faylasha Mashruuca (Project Structure)
- `app.py`: Code-ka ugu muhiimsan ee Streamlit.
- `plantvillage_model.keras`: Model-ka la tababaray ee cudurada aqoonsanaya.
- `class_names.json`: Liiska magacyada 38-ka cudur.
- `notebook.ipynb`: Code-kii lagu tababaray (Training) model-ka.

## 🚀 Sida loo kiciyo (Installation)
1. **Clone the repo:** `git clone https://github.com/YourUsername/RepoName.git`
2. **Install requirements:** `pip install streamlit tensorflow pillow numpy google-generativeai`
3. **Run App:** `streamlit run app.py`

---

## 👨‍💻 Developer & Author
**Abdifitah Ahmed Bashiir**
*Lead AI Developer & Agricultural Tech Enthusiast*

---
*Note: This project is part of an initiative to modernize farming using AI technology.*
