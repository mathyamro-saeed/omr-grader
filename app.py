import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="مصحح أوراق التظليل الذكي", layout="centered", page_icon="📝")
st.title("📝 مصحح أوراق التظليل الذكي")

# =====================================================================
# الشريط الجانبي: مفتاح الإجابة
# =====================================================================
st.sidebar.header("🗝️ مفتاح الإجابة")
options_map   = {"أ": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4}
options_label = ["أ", "ب", "ج", "د", "هـ"]

st.sidebar.subheader("قسم صح / خطأ  (5 أسئلة)")
tf_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["صح", "خطأ"], key=f"tf_{i}")
    tf_keys.append(0 if ans == "صح" else 1)

st.sidebar.subheader("قسم الاختيار من متعدد  (10 أسئلة)")
mc_keys = []
for i in range(1, 11):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["أ", "ب", "ج", "د"], key=f"mc_{i}")
    mc_keys.append(options_map[ans])

st.sidebar.subheader("قسم المزاوجة  (5 أسئلة)")
match_keys = []
for i in range(1, 6):
    ans = st.sidebar.selectbox(f"سؤال {i}", ["أ", "ب", "ج", "د", "هـ"], key=f"match_{i}")
    match_keys.append(options_map[ans])

debug_mode = st.sidebar.checkbox("🔬 وضع التشخيص", value=False)

# =====================================================================
# الإحداثيات النسبية — مُعايَرة من الورقة الفعلية بدقة
# =====================================================================
# الترتيب: الخيار 0 = أقصى اليمين (صح أو أ)، الأعداد سالبة = اتجاه RTL
LAYOUT = {
    # قسم صح/خطأ: الخيار 0=صح (يمين)، الخيار 1=خطأ (يسار)
    "tf": {
        "start_x_ratio": 0.4907,   # ← مُعايَر من الورقة الفعلية
        "step_x_ratio":  -0.0372,
        "start_y_ratio": 0.1669,
        "step_y_ratio":  0.0213,
        "n_options":   2,
        "n_questions": 5,
    },
    # قسم اختياري: الخيار 0=أ (أقصى يمين) → د (أقصى يسار)
    "mc": {
        "start_x_ratio": 0.5192,   # ← مُعايَر من الورقة الفعلية
        "step_x_ratio":  -0.0336,
        "start_y_ratio": 0.2955,
        "step_y_ratio":  0.0205,
        "n_options":   4,
        "n_questions": 10,
    },
    # قسم مزاوجة: الخيار 0=أ (أقصى يمين) → هـ (أقصى يسار)
    "match": {
        "start_x_ratio": 0.5301,   # ← مُعايَر من الورقة الفعلية
        "step_x_ratio":  -0.0329,
        "start_y_ratio": 0.5197,
        "step_y_ratio":  0.0194,
        "n_options":   5,
        "n_questions": 5,
    },
}

# =====================================================================
# دوال المساعدة
# =====================================================================

def order_points(pts):
    pts  = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    # تأكد من نسبة عرض/ارتفاع منطقية (الورقة أطول من عرضها)
    if W > H:
        W, H = H, W
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W, H)), W, H


def find_paper_contour(img_bgr):
    """يكشف أكبر مستطيل رباعي في الصورة = الورقة"""
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:10]:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area   = cv2.contourArea(c)
        if len(approx) == 4 and area > 0.15 * w * h:
            return approx
    return None


def check_bubble(thresh, cx, cy, radius=12):
    """يحسب عدد البكسلات المُظللة في منطقة دائرية"""
    h, w = thresh.shape
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    return cv2.countNonZero(thresh[y1:y2, x1:x2])


def scan_section(thresh, cfg, W, H, debug_img=None):
    """يمسح قسماً كاملاً ويُرجع قائمة الإجابات المختارة"""
    results = []
    for q in range(cfg["n_questions"]):
        cy     = int(cfg["start_y_ratio"] * H + q * cfg["step_y_ratio"] * H)
        counts = []
        for opt in range(cfg["n_options"]):
            cx = int(cfg["start_x_ratio"] * W + opt * cfg["step_x_ratio"] * W)
            counts.append(check_bubble(thresh, cx, cy))
            if debug_img is not None:
                cv2.circle(debug_img, (cx, cy), 13, (0, 220, 0), 2)
                cv2.putText(debug_img, str(opt), (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 0), 1)
        best = int(np.argmax(counts))
        results.append(best if counts[best] > 60 else -1)
    return results


# =====================================================================
# واجهة المستخدم الرئيسية
# =====================================================================
camera_file = st.camera_input("📷 صوِّر ورقة الإجابات — تأكد أن الورقة كاملة داخل الإطار")

if camera_file is not None:
    try:
        pil_img = Image.open(camera_file)
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H_orig, W_orig = img_bgr.shape[:2]

        # ── الخطوة 1: كشف حدود الورقة ────────────────────────────────
        paper = find_paper_contour(img_bgr)

        if paper is None:
            st.warning("⚠️ لم أتمكن من تحديد حدود الورقة — سأعالج الصورة كاملة.")
            warped, W, H = img_bgr.copy(), W_orig, H_orig
        else:
            st.success("✅ تم تحديد حدود الورقة بنجاح!")
            preview = img_bgr.copy()
            cv2.drawContours(preview, [paper], -1, (0, 230, 0), 4)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="الورقة المرصودة")
            warped, W, H = four_point_transform(
                img_bgr, paper.reshape(4, 2).astype("float32")
            )

        # ── الخطوة 2: تحسين التباين ثم ثنائية الصورة ─────────────────
        warped_gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe        = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        warped_eq    = clahe.apply(warped_gray)
        warped_thresh = cv2.threshold(
            warped_eq, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        # ── الخطوة 3: مسح الإجابات ───────────────────────────────────
        debug_img = warped.copy() if debug_mode else None
        tf_res    = scan_section(warped_thresh, LAYOUT["tf"],    W, H, debug_img)
        mc_res    = scan_section(warped_thresh, LAYOUT["mc"],    W, H, debug_img)
        mat_res   = scan_section(warped_thresh, LAYOUT["match"], W, H, debug_img)

        # ── الخطوة 4: التصحيح ────────────────────────────────────────
        score  = 0
        score += sum(1 for a, k in zip(tf_res,  tf_keys)    if a == k)
        score += sum(1 for a, k in zip(mc_res,  mc_keys)    if a == k)
        score += sum(1 for a, k in zip(mat_res, match_keys) if a == k)

        # ── عرض النتائج ──────────────────────────────────────────────
        st.divider()
        col_score, col_pct = st.columns(2)
        with col_score:
            st.metric("📊 الدرجة الإجمالية", f"{score} / 20")
        with col_pct:
            pct   = round(score / 20 * 100)
            grade = ("ممتاز 🌟"    if pct >= 90 else
                     "جيد جداً ✅" if pct >= 75 else
                     "جيد 👍"      if pct >= 60 else
                     "مقبول ⚠️"   if pct >= 50 else
                     "ضعيف ❌")
            st.metric("النسبة", f"{pct}%  —  {grade}")

        st.divider()

        def lbl(x, kind="mc"):
            if x == -1: return "⬜ فارغ"
            if kind == "tf": return "✔ صح" if x == 0 else "✘ خطأ"
            return options_label[x]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**صح / خطأ**")
            for i, (a, k) in enumerate(zip(tf_res, tf_keys)):
                icon = "✅" if a == k else "❌"
                st.write(f"{icon} س{i+1}: {lbl(a,'tf')}  _(صواب: {lbl(k,'tf')})_")
        with col2:
            st.markdown("**الاختياري**")
            for i, (a, k) in enumerate(zip(mc_res, mc_keys)):
                icon = "✅" if a == k else "❌"
                st.write(f"{icon} س{i+1}: {lbl(a)}  _(صواب: {lbl(k)})_")
        with col3:
            st.markdown("**المزاوجة**")
            for i, (a, k) in enumerate(zip(mat_res, match_keys)):
                icon = "✅" if a == k else "❌"
                st.write(f"{icon} س{i+1}: {lbl(a)}  _(صواب: {lbl(k)})_")

        # ── الصور المعالجة ────────────────────────────────────────────
        st.divider()
        st.subheader("🖼️ مخرجات المعالجة")
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="الورقة مُقوَّمة")
        with c2:
            st.image(warped_thresh, caption="خريطة التظليل (أبيض = مظلَّل)")

        if debug_mode and debug_img is not None:
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                     caption="🔬 مناطق الفحص — الدوائر الخضراء = مواضع القراءة")
            st.info(
                "📐 **إذا كانت الدوائر بعيدة عن الخانات:**\n\n"
                "ابحث عن `LAYOUT` في الكود وعدِّل:\n"
                "- `start_x_ratio` / `start_y_ratio`: موضع أول خانة\n"
                "- `step_x_ratio` / `step_y_ratio`: المسافة بين الخانات"
            )

    except Exception as e:
        st.error(f"⚠️ خطأ في المعالجة: {e}")
        st.info("حاول التصوير في إضاءة جيدة مع تثبيت الجوال وجعل الورقة مستوية.")

else:
    st.info("📷 وجِّه الكاميرا على ورقة الإجابات والتقط صورة لبدء التصحيح.")
    with st.expander("ℹ️ تعليمات الاستخدام"):
        st.markdown("""
**الخطوات:**
1. أدخل مفتاح الإجابة من الشريط الجانبي
2. ضع الورقة على سطح مستوٍ ذي **خلفية داكنة**
3. صوِّرها بحيث تظهر **حواف الورقة الأربعة** كاملة
4. انتظر ظهور النتيجة

**نصائح:**
- 💡 إضاءة جيدة وموحدة (تجنب الظل والوهج)
- 📄 الورقة مستوية غير مطوية
- 📐 الورقة بيضاء على خلفية داكنة = أفضل نتيجة
        """)
