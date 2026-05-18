import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="تصحيح أوراق التظليل")

st.title("تصحيح أوراق التظليل")

uploaded_file = st.file_uploader(
    "ارفع صورة الورقة",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="الصورة المرفوعة")

    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

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

    bubble_count = 0

    for c in contours:

        area = cv2.contourArea(c)

        if 200 < area < 3000:
            bubble_count += 1

    st.success(f"تم اكتشاف {bubble_count} عنصر محتمل")

    st.write("النظام يعمل بنجاح ✅")
