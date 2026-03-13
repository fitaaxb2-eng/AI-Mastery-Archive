import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
# 1. Soo rar model-ka (Hubi in magaca uu sax yahay)
model = tf.keras.models.load_model('models/happysadmodel.h5')
st.title("Saadaalinta Farxadda iyo Murugada")

# 2. Upload-ka sawirka
file = st.file_uploader("Soo geli sawirka aad rabto", type=['jpg', 'png', 'jpeg'])

if file is not None:
    # Sawirka u bedel PIL Image
    img = Image.open(file)
    st.image(img, caption="Sawirka aad soo gelisay hoos ka eeg natiijada",use_container_width=True)

    # Preprocessing
    img_array = np.array(img)
    # Haddii sawirku yahay midab (RGB), u bedel BGR sidii OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Resize ka dhig 256x256
    resize = tf.image.resize(img_bgr, (256, 256))
    # Prediction
    yhat = model.predict(np.expand_dims(resize / 255, 0))

    if yhat > 0.5:
        st.error(f"Natiijadu waa: Sad (Murugo) - Score: {yhat[0][0]:.2f}")
    else:
        st.success(f"Natiijadu waa: Happy (Farxad) - Score: {yhat[0][0]:.2f}")