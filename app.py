import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="مصحح أوراق التظليل الذكي", layout="centered")
st.title("📸 مصحح أوراق التظليل عبر الكاميرا")

# --- 1. بناء واجهة إدخال مفتاح الإجابة النموذجية في شريط جانبي ---
st.sidebar.header("🛠️ مفتاح الإجابة النموذجية")

# قسم صح وخطأ (5 أسئلة)
st.sidebar.subheader("1. قسم صح / خطأ")
tf_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"س {i} (صح/خطأ)", ["صح", "خطأ"], key=f"tf_{i}")
    tf_keys.append(0 if ans == "صح" else 1) # 0=صح، 1=خطأ

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
camera_file = st.camera_input("وجه الكاميرا الخلفية واجعل المربعات الأربعة واضحة داخل الإطار ثم التقط الصورة")

if camera_file is not None:
    try:
        # قراءة الصورة الملتقطة وتحويلها إلى مصفوفة OpenCV
        image = Image.open(camera_file)
        img = np.array(image)
        original = img.copy()
        
        # تحويل الصورة إلى رمادي وتطبيق فلتر التباين الحاد
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        
        # البحث عن الأشكال الخارجية في الصورة
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_squares = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if 80 < area < 20000:
                x, y, w, h = cv2.boundingRect(c)
                ratio = w / float(h)
                
                # حساب ميزة Solidity و Extent للتأكد أن الشكل مربع مصمت
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / float(hull_area) if hull_area > 0 else 0
                extent = area / float(w * h)
                
                if 0.8 <= ratio <= 1.2 and solidity > 0.85 and extent > 0.82:
                    detected_squares.append((x + w//2, y + h//2))
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # التحقق من رصد الزوايا الأربعة الحقيقية حصرياً
        if len(detected_squares) == 4:
            st.success("تم قفل زوايا الورقة الأربعة بنجاح! ✅")
            
            pts_array = np.array(detected_squares, dtype="float32")
            doc_rect = order_points(pts_array)
            
            maxWidth, maxHeight = 500, 700  
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # حساب مصفوفة التحويل وقص الصورة
            M = cv2.getPerspectiveTransform(doc_rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            # تحويل الورقة المقصوصة لأبيض وأسود حاد لقراءة التضليل الداخلي
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            student_score = 0
            details = []

            # دالة فحص التضليل المصلحة والمحمية من أخطاء النطاق السابقة
            def check_zone(total_options, start_x, start_y, step_x, step_y, q_idx):
                max_pixels = 0
                selected_idx = -1
                for opt in range(total_options):
                    # استخدام المتغيرات الصحيحة الممررة للدالة لمنع خطأ الـ NameError
                    x = int(start_x + (opt * step_x))
                    y = int(start_y + (q_idx * step_y))
                    
                    if 10 <= x <= 490 and 10 <= y <= 690:
                        roi = warped_thresh[y-9:y+9, x-9:x+9]
                        pixel_count = cv2.countNonZero(roi)
                        if pixel_count > max_pixels and pixel_count > 90: 
                            max_pixels = pixel_count
                            selected_idx = opt
                return selected_idx

            st.subheader("📊 النتيجة التفصيلية للورقة:")
            
            # 1. قسم صح/خطأ (5 أسئلة)
            tf_results = []
            startX_TF, startY_TF, stepX_TF, stepY_TF = 265, 205, -30, 24
            for i in range(5):
                ans = check_zone(2, startX_TF, startY_TF, stepX_TF, stepY_TF, i)
                tf_results.append(ans)
                if ans == tf_keys[i]: student_score += 1
            details.append(f"**إجابات صح/خطأ:** {[ 'ص' if x==0 else ('خ' if x==1 else 'فارغ') for x in tf_results]}")

            # 2. قسم الاختيار من متعدد (10 أسئلة)
            mc_results = []
            startX_MC, startY_MC, stepX_MC, stepY_MC = 288, 395, -25, 21
            for i in range(10):
                ans = check_zone(4, startX_MC, startY_MC, stepX_MC, stepY_MC, i)
                mc_results.append(ans)
                if ans == mc_keys[i]: student_score += 1
            details.append(f"**إجابات الاختياري:** {[options_labels[x] if x!=-1 else 'فارغ' for x in mc_results]}")

            # 3. قسم المزاوجة (5 أسئلة)
            match_results = []
            startX_Match, startY_Match, stepX_Match, stepY_Match = 300, 605, -25, 21
            for i in range(5):
                ans = check_zone(5, startX_Match, startY_Match, stepX_Match, stepY_Match, i)
                match_results.append(ans)
                if ans == match_keys[i]: student_score += 1
            details.append(f"**إجابات المزاوجة:** {[options_labels[x] if x!=-1 else 'فارغ' for x in match_results]}")

            # إظهار النتيجة الكلية
            st.metric(label="الدرجة الإجمالية للطالب", value=f"{student_score} / 20")
            
            for det in details:
                st.write(det)
                
            # عرض الصور المعالجة
            st.subheader("📷 مخرجات المعالجة الرقمية:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(warped, caption="الورقة مقصوصة ومستقيمة")
            with col2:
                st.image(warped_thresh, caption="تحليل نقاط التضليل الداخلي")

        else:
            st.error(f"❌ لم يتم رصد المربعات الأربعة الحقيقية بشكل صحيح (تم العثور على {len(detected_squares)} معالم).")
            st.info("💡 يرجى التأكد من أن المربعات السوداء الأربعة تظهر بالكامل داخل الصورة الملتقطة.")
            st.image(original, caption="المعالم المرصودة حالياً")
            
    except Exception as e:
        st.error(f"⚠️ حدث خطأ غير متوقع أثناء المعالجة البرمجية: {e}")
        st.info("الرجاء المحاولة مرة أخرى والتقاط الصورة في بيئة ذات إضاءة ممتازة وثابتة.")

else:
    st.info("💡 في انتظار التقاط الصورة لبدء عملية التصحيح التلقائي...")
