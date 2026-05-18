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

    img = np.array(image)

    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    page = None

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for c in contours:

        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.02 * peri,
            True
        )

        if len(approx) == 4:
            page = approx
            break

    draw = original.copy()

    if page is not None:

        cv2.drawContours(
            draw,
            [page],
            -1,
            (0,255,0),
            8
        )

        st.success("تم اكتشاف الورقة بنجاح ✅")

    else:

        st.error("لم يتم اكتشاف حدود الورقة")

    st.image(draw)

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    contours2, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circles = 0

    for c in contours2:

        area = cv2.contourArea(c)

        x,y,w,h = cv2.boundingRect(c)

        ratio = w / float(h)

        if (
            150 < area < 2000
            and 0.7 < ratio < 1.3
        ):
            circles += 1

    st.info(f"تم اكتشاف {circles} دائرة محتملة")
