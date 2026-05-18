import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="مصحح أوراق التظليل المباشر", layout="centered")
st.title("📸 مصحح أوراق التظليل عبر الكاميرا المباشرة")

# --- 1. بناء واجهة إدخال مفتاح الإجابة النموذجية في شريط جانبي ---
st.sidebar.header("🛠️ مفتاح الإجابة النموذجية")

# قسم صح وخطأ (5 أسئلة)
st.sidebar.subheader("1. قسم صح / خطأ")
tf_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"س {i} (صح/خطأ)", ["صح", "خطأ"], key=f"tf_{i}")
    tf_keys.append(0 if ans == "صح" else 1)

# قسم الاختيار من متعدد (10 أسئلة)
st.sidebar.subheader("2. قسم الاختيار من متعدد")
mc_keys = []
options_map = {"أ": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4}
options_labels = ["أ", "ب", "ج", "د", "هـ"]
for i in range(1, 11):
    ans = st.sidebar.selectbox(f"س {i} (اختياري)", ["أ", "ب", "ج", "د"], key=f"mc_{i}")
    mc_keys.append(options_map[ans])

# قسم المزاوجة (5 أسئلة)
st.sidebar.subheader("3. قسم المزاوجة")
match_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"س {i} (مزاوجة)", ["أ", "ب", "ج", "د", "هـ"], key=f"match_{i}")
    match_keys.append(options_map[ans])


# --- 2. دالة ترتيب نقاط زوايا الورقة برمجياً ---
def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # أعلى اليسار
    rect[2] = pts[np.argmax(s)]  # أسفل اليمين
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # أعلى اليمين
    rect[3] = pts[np.argmax(diff)] # أسفل اليسار
    return rect


# --- 3. تشغيل الكاميرا المباشرة وبدء المعالجة ---
# هذه الأداة ستفتح الكاميرا فوراً داخل المتصفح
camera_file = st.camera_input("وجه الكاميرا الخلفية نحو الورقة بشكل مستقيم وثابت ثم التقط الصورة")

if camera_file is not None:
    # قراءة الصورة الملتقطة وتحويلها إلى مصفوفة OpenCV
    image = Image.open(camera_file)
    img = np.array(image)
    original = img.copy()
    
    # تحويل الصورة إلى رمادي وتنعيمها
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)
    
    # البحث عن محيط الورقة الخارجية
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    page = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 30000: # تقليل العتبة لتناسب أحجام الكاميرا المختلفة
            page = approx
            break
            
    if page is not None:
        st.success("تم التقاط وتحليل حدود الورقة بنجاح! ✅")
        
        # قص وتعديل زاوية الورقة هندسياً
        doc_rect = order_points(page)
        maxWidth, maxHeight = 500, 700  
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(doc_rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        # تحويل الورقة المقصوصة لأبيض وأسود حاد تلقائي (Otsu)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # حقول حفظ النتيجة
        student_score = 0
        details = []

        # دالة فحص التضليل المحسنة
        def check_zone(total_options, start_x, start_y, step_x, step_y, q_idx):
            max_pixels = 0
            selected_idx = -1
            for opt in range(total_options):
                x = int(start_x + (opt * step_x))
                y = int(start_y + (q_idx * step_y))
                
                # منطقة الفحص الدائرية
                roi = warped_thresh[y-9:y+9, x-9:x+9]
                pixel_count = cv2.countNonZero(roi)
                
                if pixel_count > max_pixels and pixel_count > 80: 
                    max_pixels = pixel_count
                    selected_idx = opt
            return selected_idx

        # 1. فحص قسم صح/خطأ (5 أسئلة)
        st.subheader("📊 النتيجة الفورية للورقة الملتقطة:")
        tf_results = []
        for i in range(5):
            ans = check_zone(2, 385, 235, -45, 33, i)
            tf_results.append(ans)
            if ans == tf_keys[i]: student_score += 1
        details.append(f"**إجابات صح/خطأ:** {[ 'ص' if x==0 else ('خ' if x==1 else 'فارغ') for x in tf_results]}")

        # 2. فحص قسم الاختيار من متعدد (10 أسئلة)
        mc_results = []
        for i in range(10):
            ans = check_zone(4, 222, 235, -32, 33, i)
            mc_results.append(ans)
            if ans == mc_keys[i]: student_score += 1
        details.append(f"**إجابات الاختياري:** {[options_labels[x] if x!=-1 else 'فارغ' for x in mc_results]}")

        # 3. فحص قسم المزاوجة (5 أسئلة)
        match_results = []
        for i in range(5):
            ans = check_zone(5, 82, 235, -28, 33, i)
            match_results.append(ans)
            if ans == match_keys[i]: student_score += 1
        details.append(f"**إجابات المزاوجة:** {[options_labels[x] if x!=-1 else 'فارغ' for x in match_results]}")

        # إظهار النتيجة الكلية بارزة
        st.metric(label="الدرجة الإجمالية للطالب", value=f"{student_score} / 20")
        
        for det in details:
            st.write(det)
            
        # عرض نتائج المعالجة الهندسية
        st.subheader("📷 مخرجات المعالجة الرقمية:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(warped, caption="الورقة مقصوصة ومستقيمة")
        with col2:
            st.image(warped_thresh, caption="تحليل الحبر والتضليل")

    else:
        st.warning("⚠️ يرجى الضغط على زر التقاط الصورة (Take Photo) بعد توجيه الكاميرا بشكل جيد ليتم بدء التصحيح.")
