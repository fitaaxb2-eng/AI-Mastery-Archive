import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import google.generativeai as genai

# --- STEP 1: CONFIGURATION (Dejinta Bogga) ---
st.set_page_config(page_title="PlantVillage AI", page_icon="🌿")

# --- STEP 2: API SETUP (Xiriirka Gemini) ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except:
    st.error("❌ API Key lama helin!")
    st.stop()


# --- STEP 3: MODEL PICKER (Doorashada Model-ka) ---
@st.cache_resource
def get_best_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return "gemini-1.5-flash" if "models/gemini-1.5-flash" in models else models[0].replace("models/", "")
    except:
        return "gemini-1.5-flash"


SELECTED_MODEL = get_best_model()

# --- STEP 4: EXPERT RULES (Xeerarka AI-ga) ---
# --- STEP 4: EXPERT RULES (Xeerarka AI-ga ee la cusboonaysiiyey) ---
SYSTEM_PROMPT = """
Magacaaga waa 'Khabiirka Dhirta'. Waxaad tahay khabiir ku takhasusay caafimaadka dhirta iyo beeraha. 
Waxaa ku dhisay oo masuul ka ah Abdifitah Ahmed Bashiir.

RAAC XEERARKAN:
1. LUQADDA: Markasta ku jawaab Af-Soomaali cad oo habaysan.
2. SALAANTA & IS-WARAYSIGA: Haddii lagu salamo (sida: Asc, Iska waran, Sidee tahay), si caadi ah oo asluub leh u jawaab (sida: "Waan fiicanahay, mahadsanid. Sidee ku caawin karaa?").
3. DHISMAHA AI-GA: Haddii lagu weydiiyo "Yaa ku sameeyay?", "Yaa ku dhisay?", ama "Yaa masuul ka ah?", markasta dheh: "Waxaa i dhisay oo masuul iga ah Abdifitah Ahmed Bashiir."
4. WEPROMPT-KA SU'AALAHA: Haddii user-ku yiraahdo "Su'aal baan rabaa inaan ku weydiiyo" ama wax la mid ah, ugu jawaab: "Haa, soo dhawoow! Isaa weydii wax kasta oo ku saabsan dhirta iyo beeraha, waan kugu caawinayaa."
5. MAWDUUCA DHIRTA: Wixii su'aal ah oo ka baxsan salaanta, aqoonsigaaga (Creator), iyo dhirta/beeraha, si asluub leh u dheh: "Raali ahow, anigu waxaan ahay khabiir ku takhasusay dhirta iyo beeraha oo kaliya, su'aashaas aqoon uma laha."
6. IS-HUBIN (Clarification): Haddii su'aashu qasan tahay ama aad fahmi weydo, weydii: "Ma wax ku saabsan dhirta iyo beeraha ayaad ka wadaa mise wax kale haku degdegin raali ahow aniga waxan ahy khabir ku takhusay dhirta iska hubi first haduu shaki ku galo?"
7. TALOOYINKA CUDURADA: Marka uu model-ku cudur helo, sii sharaxaad kooban oo ku saabsan cudurkan iyo 3 talo oo wax ku ool ah oo beeralaydu isticmaali karaan.
"""
# --- STEP 5: INITIALIZE GEMINI (Diyaarinta Gemini) ---
model_gemini = genai.GenerativeModel(
    model_name=SELECTED_MODEL,
    system_instruction=SYSTEM_PROMPT
)


# --- STEP 6: LOAD IMAGE MODEL (Soo rarista Model-ka Keras) ---
@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model("plantvillage_model.keras", compile=False)


model_keras = load_plant_model()

# --- STEP 7: LOAD CLASS NAMES (Magacyada Cudurada) ---
try:
    with open('class_names.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except:
    st.error("❌ File-ka 'class_names.json' lama helin.")
    st.stop()

# --- STEP 8: CHAT MEMORY (Kaydka Sheekada) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STEP 9: SIDEBAR SETTINGS (Dhinaca Control-ka) ---
with st.sidebar:
    st.header("Settings")
    st.write(f"🤖 AI Model: `{SELECTED_MODEL}`")
    if st.button("Tirtir Sheekada"):
        st.session_state.messages = []
        if "analysis_done" in st.session_state: del st.session_state.analysis_done
        st.rerun()


# --- STEP 10: FILE UPLOADER (Meesha Sawirka laga ridayo) ---
st.title("🌿 PlantVillage AI")
file = st.file_uploader("Soo geli sawirka caleenta...", type=["jpg", "png", "jpeg"])

current_file_name = file.name if file else None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if st.session_state.last_uploaded_file != current_file_name:
    st.session_state.messages = []

    if "analysis_done" in st.session_state:
        del st.session_state.analysis_done
    st.session_state.last_uploaded_file = current_file_name
    st.rerun()


# --- STEP 11: IMAGE PROCESSING (Diyaarinta Sawirka) ---
if file:
    img = Image.open(file).resize((224, 224))
    st.image(img, caption="Sawirka la soo rartay", width=300)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # --- STEP 12: PREDICTION (Baarista Cudurka) ---
    with st.spinner("AI-ga ayaa baaraya..."):
        predictions = model_keras(img_array, training=False)
        res = CLASS_NAMES[np.argmax(predictions[0])]
        conf = np.max(predictions[0])

    # --- STEP 13: SHOW RESULTS (Muujinta Natiijada) ---
    st.divider()

    # Halkii aad hal sadar ka dhigi lahayd, u kala jibar sidan:
    if "Healthy" in res:
        st.success(f"Natiijada: {res}")
    else:
        st.error(f"Natiijada: {res}")

    st.info(f"🎯 Kalsoonida: {conf:.2%}")

    # --- STEP 14: AUTO AI SUMMARY (Warbixinta Tooska ah) ---
    if "analysis_done" not in st.session_state:
        with st.spinner("Khabiirka ayaa qoraya warbixin..."):
            prompt = f"Cudurka la helay waa {res}. Sharaxaad kooban iyo 3 talo sii beeralayda."
            response = model_gemini.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.session_state.analysis_done = True
else:
    st.info("Fadlan sawir soo geli si aad u hesho xog.")

# --- STEP 15: CHAT INTERFACE (Sheekada Khabiirka) ---
st.divider()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Weydii khabiirka..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = model_gemini.generate_content(user_input)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})