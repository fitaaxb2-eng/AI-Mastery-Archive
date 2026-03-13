import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Ciwaanka App-ka
st.title("🍅 Tomato Health Check")

# 2. Soo rary moodelka (Hubi inuu file-ka model-ka magacan leeyahay)
model = tf.keras.models.load_model("models/tomato_leaf_model.h5")

# 3. Upload-ka sawirka
file = st.file_uploader("Soo geli sawirka caleenta", type=["jpg", "png", "jpeg"])

if file:
    # Muuji sawirka oo loo habeeyey 224x224
    img = Image.open(file).resize((224, 224))
    st.image(img, caption="Sawirka aad soo gelisay")

    # 4. Saadaalinta (Prediction)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # 5. Natiijada (0 = Disease, 1 = Health)
    if prediction > 0.5:
        st.success(f"Natiijada: Caafimaad (Health_Tomato)")
    else:
        st.error(f"Natiijada: Cudur (Disease_Tomato)")

    st.write(f"Kalsoonida: {prediction if prediction > 0.5 else 1 - prediction:.2%}")