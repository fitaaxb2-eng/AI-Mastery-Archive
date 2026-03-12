#  Bangalore House Price Prediction (BHP)

This repository contains an end-to-end Machine Learning project for predicting house prices in Bangalore. The project demonstrates a full pipeline from data cleaning and model building to deployment using both **Flask** and **Streamlit**.

---

## 🇬🇧 English Description

###  Project Overview
The goal of this project is to build a regression model to predict property prices based on location, square footage, bedrooms (BHK), and bathrooms. The project is structured to mimic real-world software development practices, separating the model, server, and frontend logic.

###  Repository Structure
The project is divided into 4 main folders:

1.  **`modelbuilding`**:
    *   **Content:** Raw dataset (`.csv`) and Jupyter Notebook (`.ipynb`).
    *   **Process:** Data Cleaning, Feature Engineering, Outlier Removal, and Model Selection using **GridSearchCV** (Linear Regression, Lasso, Decision Tree).
2.  **`server`**:
    *   **Backend:** Built with **Python Flask**.
    *   **Artifacts:** Contains the `artifacts` folder with the trained `saved_model.pickle` and `columns.json`.
    *   **Logic:** `server.py` handles the API requests, and `util.py` manages loading the model artifacts.
3.  **`webflaskversion`**:
    *   **Frontend:** Standard web interface using HTML, CSS, and JavaScript.
    *   **Function:** Connects to the Flask server to fetch predictions via HTTP calls.
4.  **`streamlitversion`**:
    *   **Frontend (Modern):** A standalone **Streamlit** application.
    *   **Portable:** Contains its own copy of `pickle` and `json` files for easy deployment without needing a separate backend server run.

###  How to Run

**Option 1: Flask Web App**
1. Navigate to the server folder: `cd server`
2. Run the server: `python server.py`
3. Open the `app.html` file inside `webflaskversion` in your browser.

**Option 2: Streamlit App (Recommended/Easiest)**
1. Navigate to the streamlit folder: `cd streamlitversion`
2. Run the app: `streamlit run streamlitapp.py`

---

## 🇸🇴 Af-Soomaali: Faahfaahinta Mashruuca

###  Hordhac
Mashruucan waa codsi (application) dhamaystiran oo lagu saadaalinayo qiimaha guryaha magaalada Bangalore. Wuxuu daboolayaa dhammaan heerarka *Data Science*, laga soo bilaabo nadiifinta xogta ilaa laga gaarayo in dadka loo soo bandhigo (Deployment).

###  Qaab-dhismeedka (Folders)
Mashruucan wuxuu u qaybsan yahay 4 qaybood oo waaweyn:

1.  **`modelbuilding`**: Halkan waxaa lagu dhisay Model-ka. Waxaa ku jira `csv` file-ka iyo `Notebook` (.ipynb) oo laga sameeyay *Data Cleaning* iyo *Model Training*.
2.  **`server`**: Waa *Backend-ka* oo ku shaqaynaya **Flask**. Waxaa ku jira `artifacts` (pickle & json files). `server.py` ayaa maamulaya codsiyada.
3.  **`webflaskversion`**: Waa wajiga hore (Frontend) ee HTML/CSS/JS ah, kaas oo la xiriiraya server-ka Flask.
4.  **`streamlitversion`**: Waa version fudud oo ku shaqaynaya **Streamlit**. Waa folder gooni u taagan oo wata model-kiisa, aadna way u fududahay in la kiciyo.

---

## 👤 Author & Credits

**Author / Qoraa:**
*   **[Abdifitah Ahmed Bashiir]** 
*   *Data Science & Machine Learning Enthusiast*

**Credits / Mahadcelin:**
*   This project is based on the **Codebasics** Real World Data Science Bootcamp.
*   Special thanks to the open-source community.

---
