import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="مصحح أوراق التظليل الذكي", layout="centered", page_icon="📝")
st.title("📝 مصحح أوراق التظليل الذكي")

# =====================================================================
# الشريط الجانبي: مفتاح الإجابة النموذجية
# =====================================================================
st.sidebar.header("🗝️ مفتاح الإجابة")
options_map   = {"أ": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4}
options_rev   = {v: k for k, v in options_map.items()}
options_label = ["أ", "ب", "ج", "د", "هـ"]

st.sidebar.subheader("قسم صح / خطأ")
tf_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["صح", "خطأ"], key=f"tf_{i}")
    tf_keys.append(0 if ans == "صح" else 1)

st.sidebar.subheader("قسم الاختيار من متعدد")
mc_keys = []
for i in range(1, 11):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["أ", "ب", "ج", "د"], key=f"mc_{i}")
    mc_keys.append(options_map[ans])

st.sidebar.subheader("قسم المزاوجة")
match_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["أ", "ب", "ج", "د", "هـ"], key=f"match_{i}")
    match_keys.append(options_map[ans])

# وضع التشخيص: يُظهر الـ ROI على الصورة
debug_mode = st.sidebar.checkbox("🔬 وضع التشخيص (عرض مناطق الفحص)", value=False)

# =====================================================================
# الإحداثيات النسبية لكل قسم (نسبة من 0 إلى 1 من أبعاد الورقة)
# =====================================================================
# هذه القيم مبنية على التصميم المعياري للورقة.
# عدّلها من وضع التشخيص إذا كانت ورقتك مختلفة.
LAYOUT = {
    # قسم صح/خطأ: عمودان (صح=0، خطأ=1)، 5 أسئلة
    "tf": {
        "start_x_ratio": 0.53,   # نسبة بداية أول دائرة أفقياً
        "step_x_ratio":  -0.06,  # المسافة بين الدوائر أفقياً (سالب = من يمين لشمال)
        "start_y_ratio": 0.293,  # نسبة بداية أول سؤال عمودياً
        "step_y_ratio":  0.034,  # المسافة بين الأسئلة عمودياً
        "n_options": 2,
        "n_questions": 5,
    },
    # قسم الاختيار من متعدد: 4 خيارات، 10 أسئلة
    "mc": {
        "start_x_ratio": 0.576,
        "step_x_ratio":  -0.05,
        "start_y_ratio": 0.564,
        "step_y_ratio":  0.030,
        "n_options": 4,
        "n_questions": 10,
    },
    # قسم المزاوجة: 5 خيارات، 5 أسئلة
    "match": {
        "start_x_ratio": 0.60,
        "step_x_ratio":  -0.05,
        "start_y_ratio": 0.864,
        "step_y_ratio":  0.030,
        "n_options": 5,
        "n_questions": 5,
    },
}

# =====================================================================
# دوال المساعدة
# =====================================================================
def order_points(pts):
    """ترتيب نقاط المستطيل: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار"""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]    # أعلى اليسار  (أصغر مجموع)
    rect[2] = pts[np.argmax(s)]    # أسفل اليمين  (أكبر مجموع)
    rect[1] = pts[np.argmin(diff)] # أعلى اليمين  (أصغر فرق)
    rect[3] = pts[np.argmax(diff)] # أسفل اليسار  (أكبر فرق)
    return rect


def four_point_transform(image, pts):
    """تقويم الورقة perspectively وتحويلها إلى مستطيل مستوٍ"""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxW    = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH    = max(int(heightA), int(heightB))

    dst = np.array([
        [0,        0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0,        maxH - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH)), maxW, maxH


def detect_paper_corners(img_bgr):
    """
    يبحث عن المربعات/الدوائر السوداء الأربعة في الزوايا.
    يُرجع مصفوفة شكلها (4,2) أو None إذا فشل.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)

    # إزالة الضوضاء
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img_bgr.shape[:2]
    min_area = (w_img * h_img) * 0.0005   # 0.05 % من مساحة الصورة
    max_area = (w_img * h_img) * 0.04     # 4   % من مساحة الصورة

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        ratio     = w / float(h)
        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        extent   = area / float(w * h)

        if 0.70 <= ratio <= 1.35 and solidity > 0.80 and extent > 0.70:
            cx, cy = x + w // 2, y + h // 2
            candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return None, candidates

    # اختر أقرب 4 نقاط لزوايا الصورة
    corners_target = [
        (0, 0),           # أعلى اليسار
        (w_img, 0),       # أعلى اليمين
        (w_img, h_img),   # أسفل اليمين
        (0, h_img),       # أسفل اليسار
    ]
    chosen = []
    used   = set()
    for tx, ty in corners_target:
        best_dist = float("inf")
        best_idx  = -1
        for idx, (cx, cy, _) in enumerate(candidates):
            if idx in used:
                continue
            d = (cx - tx) ** 2 + (cy - ty) ** 2
            if d < best_dist:
                best_dist = d
                best_idx  = idx
        if best_idx != -1:
            used.add(best_idx)
            chosen.append(candidates[best_idx][:2])

    if len(chosen) == 4:
        return np.array(chosen, dtype="float32"), candidates
    return None, candidates


def check_bubble(warped_thresh, cx, cy, radius=10):
    """
    يفحص منطقة دائرية حول (cx, cy) في الصورة الثنائية.
    يُرجع عدد البكسلات المُعتمة (المُظللة).
    """
    h, w = warped_thresh.shape
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    roi = warped_thresh[y1:y2, x1:x2]
    return cv2.countNonZero(roi)


def scan_section(warped_thresh, layout_cfg, W, H, debug_img=None):
    """
    يمسح قسماً كاملاً ويُرجع قائمة الإجابات المختارة.
    layout_cfg: قاموس الإحداثيات النسبية للقسم.
    W, H: أبعاد الورقة المُقوَّمة.
    debug_img: إذا مُرِّرت صورة، يرسم عليها مناطق الفحص.
    """
    results = []
    for q in range(layout_cfg["n_questions"]):
        counts   = []
        cy = int(layout_cfg["start_y_ratio"] * H + q * layout_cfg["step_y_ratio"] * H)
        for opt in range(layout_cfg["n_options"]):
            cx = int(layout_cfg["start_x_ratio"] * W + opt * layout_cfg["step_x_ratio"] * W)
            cnt = check_bubble(warped_thresh, cx, cy)
            counts.append(cnt)

            if debug_img is not None:
                color = (0, 200, 0)
                cv2.circle(debug_img, (cx, cy), 10, color, 2)

        max_cnt = max(counts)
        if max_cnt > 80:   # عتبة الحد الأدنى للتظليل
            results.append(int(np.argmax(counts)))
        else:
            results.append(-1)  # لم يُظلَّل
    return results


# =====================================================================
# واجهة الكاميرا والمعالجة الرئيسية
# =====================================================================
camera_file = st.camera_input("📷 التقط صورة لورقة الإجابات (تأكد من ظهور الزوايا الأربعة)")

if camera_file is not None:
    try:
        pil_image = Image.open(camera_file)
        img_rgb   = np.array(pil_image)
        img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        display   = img_bgr.copy()

        # --- الخطوة 1: اكتشاف الزوايا ---
        corners, all_candidates = detect_paper_corners(img_bgr)

        # ارسم جميع المرشحين بالأزرق
        for cx, cy, _ in all_candidates:
            cv2.circle(display, (cx, cy), 12, (255, 100, 0), 3)

        if corners is None:
            st.error(f"❌ لم يتم رصد زوايا الورقة. عدد المرشحين: {len(all_candidates)}")
            st.info("💡 تأكد من:\n- وضوح المربعات/الدوائر السوداء الأربعة\n- إضاءة جيدة وخلفية فاتحة\n- عدم وجود انعكاسات ضوئية")
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="المرشحون المرصودون (أزرق)")
            st.stop()

        # ارسم الزوايا المختارة بالأخضر
        for cx, cy in corners:
            cv2.circle(display, (int(cx), int(cy)), 15, (0, 220, 0), -1)
        cv2.polylines(display, [corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        st.success("✅ تم قفل زوايا الورقة الأربعة!")
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="الزوايا المرصودة")

        # --- الخطوة 2: تقويم الورقة ---
        warped, W, H = four_point_transform(img_bgr, corners)
        warped_rgb   = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        warped_gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_thresh = cv2.threshold(
            warped_gray, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        # --- الخطوة 3: المسح ---
        debug_img = warped.copy() if debug_mode else None

        tf_results    = scan_section(warped_thresh, LAYOUT["tf"],    W, H, debug_img)
        mc_results    = scan_section(warped_thresh, LAYOUT["mc"],    W, H, debug_img)
        match_results = scan_section(warped_thresh, LAYOUT["match"], W, H, debug_img)

        # --- الخطوة 4: التصحيح ---
        score = 0
        score += sum(1 for a, k in zip(tf_results,    tf_keys)    if a == k)
        score += sum(1 for a, k in zip(mc_results,    mc_keys)    if a == k)
        score += sum(1 for a, k in zip(match_results, match_keys) if a == k)

        # --- عرض النتائج ---
        st.subheader("📊 النتائج")
        st.metric("الدرجة الإجمالية", f"{score} / 20")

        def fmt_tf(lst):
            return [("ص" if x == 0 else ("خ" if x == 1 else "⬜")) for x in lst]

        def fmt_mc(lst):
            return [(options_label[x] if 0 <= x <= 4 else "⬜") for x in lst]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**صح/خطأ**")
            for i, (a, k) in enumerate(zip(tf_results, tf_keys)):
                icon = "✅" if a == k else "❌"
                st.write(f"{icon} س{i+1}: {fmt_tf([a])[0]} (صواب: {'ص' if k==0 else 'خ'})")
        with col2:
            st.write("**الاختياري**")
            for i, (a, k) in enumerate(zip(mc_results, mc_keys)):
                icon = "✅" if a == k else "❌"
                ans_lbl = options_label[a] if a != -1 else "⬜"
                st.write(f"{icon} س{i+1}: {ans_lbl} (صواب: {options_label[k]})")
        with col3:
            st.write("**المزاوجة**")
            for i, (a, k) in enumerate(zip(match_results, match_keys)):
                icon = "✅" if a == k else "❌"
                ans_lbl = options_label[a] if a != -1 else "⬜"
                st.write(f"{icon} س{i+1}: {ans_lbl} (صواب: {options_label[k]})")

        # --- عرض الصور ---
        st.subheader("🖼️ مخرجات المعالجة")
        c1, c2 = st.columns(2)
        with c1:
            st.image(warped_rgb, caption="الورقة مُقوَّمة")
        with c2:
            st.image(warped_thresh, caption="تحليل التظليل")

        if debug_mode and debug_img is not None:
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                     caption="🔬 مناطق الفحص (دوائر خضراء)")
            st.info(
                "📐 إذا كانت الدوائر بعيدة عن خانات الإجابة، عدِّل قيم LAYOUT في الكود:\n"
                "- start_x_ratio / start_y_ratio: موضع أول خانة (نسبة من العرض/الارتفاع)\n"
                "- step_x_ratio / step_y_ratio: المسافة بين الخانات"
            )

    except Exception as e:
        st.error(f"⚠️ خطأ أثناء المعالجة: {e}")
        st.info("الرجاء التصوير في ضوء جيد وتأكد من ظهور كل الورقة.")

else:
    st.info("📷 وجِّه الكاميرا على ورقة الإجابات والتقط صورة لبدء التصحيح.")

    with st.expander("ℹ️ كيفية الاستخدام"):
        st.markdown("""
        **الخطوات:**
        1. أدخل مفتاح الإجابة في الشريط الجانبي
        2. ضع الورقة على خلفية فاتحة
        3. التقط الصورة بحيث تظهر الزوايا الأربعة بوضوح
        4. انتظر ظهور النتيجة
        
        **نصائح للحصول على أفضل النتائج:**
        - إضاءة جيدة ومتساوية
        - لا توجد انعكاسات أو ظلال
        - الورقة مستوية وليست مطوية
        - المربعات السوداء في الزوايا واضحة تماماً
        
        **وضع التشخيص:**
        - فعّله من الشريط الجانبي إذا كانت النتائج غير دقيقة
        - ستظهر دوائر خضراء على مناطق الفحص
        - عدِّل قيم LAYOUT في الكود إذا كانت الدوائر بعيدة عن الخانات
        """)
