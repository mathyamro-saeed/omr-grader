import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("تصحيح أوراق التظليل")

uploaded_file = st.file_uploader(
    "ارفع صورة الورقة",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circles = 0

    for c in contours:

        area = cv2.contourArea(c)

        if 100 < area < 2000:
            circles += 1

    st.image(image)

    st.success(f"تم اكتشاف {circles} دائرة")
