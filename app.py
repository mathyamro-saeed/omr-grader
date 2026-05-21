import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="مصحح أوراق التظليل الذكي", layout="centered", page_icon="📝")
st.title("📝 مصحح أوراق التظليل الذكي")

# =====================================================================
# الشريط الجانبي
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
# دوال المساعدة
# =====================================================================

def order_points(pts):
    pts  = pts.reshape(4, 2).astype("float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect    = order_points(pts)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W > H:          # الورقة دائماً أطول من عرضها
        W, H = H, W
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W, H)), W, H


def find_paper_contour(img_bgr):
    h, w  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:10]:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.15 * w * h:
            return approx
    return None


def detect_bubbles(warped_gray):
    """
    يكشف كل الدوائر في الورقة المُقوَّمة باستخدام HoughCircles.
    يُرجع مصفوفة (N, 3): x, y, r
    """
    blur = cv2.GaussianBlur(warped_gray, (7, 7), 0)
    H, W = warped_gray.shape

    # نصف قطر الدوائر نسبةً إلى عرض الورقة (الدوائر صغيرة = ~2% من العرض)
    min_r = max(6,  int(W * 0.012))
    max_r = max(18, int(W * 0.030))
    min_dist = max(12, int(W * 0.025))

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=min_dist,
        param1=55, param2=22,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return np.array([])
    return np.round(circles[0]).astype(int)


def find_section_separators(warped_gray):
    """
    يجد الخطوط الأفقية الفاصلة بين الأقسام الثلاثة.
    يُرجع قائمة y-positions مرتبة.
    """
    H, W = warped_gray.shape
    # Canny + Hough lines
    edges = cv2.Canny(warped_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(W*0.4),
                            minLineLength=int(W*0.4), maxLineGap=20)
    if lines is None:
        return []

    y_positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:          # خط أفقي
            y_mid = (y1 + y2) // 2
            # تجاهل الخطوط في الحواف
            if 0.05 * H < y_mid < 0.95 * H:
                y_positions.append(y_mid)

    if not y_positions:
        return []

    # دمج الخطوط المتقاربة (±15px)
    y_positions.sort()
    merged = [y_positions[0]]
    for y in y_positions[1:]:
        if y - merged[-1] > 15:
            merged.append(y)
        else:
            merged[-1] = (merged[-1] + y) // 2

    return merged


def assign_bubbles_to_sections(circles, separators, H):
    """
    يقسم الدوائر على الأقسام الثلاثة بناءً على y-position والفواصل.
    يُرجع 3 قوائم: tf_circles, mc_circles, match_circles
    """
    if len(separators) >= 2:
        sep = sorted(separators)
        # نأخذ أول فاصلين داخل نطاق 10-90% من الارتفاع
        inner = [s for s in sep if 0.10 * H < s < 0.90 * H]
        if len(inner) >= 2:
            y1, y2 = inner[0], inner[1]
        elif len(inner) == 1:
            y1 = inner[0]
            y2 = int(H * 0.72)
        else:
            y1 = int(H * 0.38)
            y2 = int(H * 0.72)
    else:
        # قيم افتراضية نسبية إذا لم تُكتشف الفواصل
        y1 = int(H * 0.38)
        y2 = int(H * 0.72)

    tf_c    = [c for c in circles if c[1] < y1]
    mc_c    = [c for c in circles if y1 <= c[1] < y2]
    match_c = [c for c in circles if c[1] >= y2]
    return tf_c, mc_c, match_c, y1, y2


def grid_from_circles(circles, n_questions, n_options):
    """
    يُرتّب الدوائر في شبكة (n_questions × n_options) تلقائياً.
    - يجمّع الدوائر أفقياً في صفوف (أسئلة)
    - يُرتّب الأعمدة من اليمين لليسار (RTL)
    يُرجع قائمة n_questions × n_options من مراكز الدوائر أو None
    """
    if len(circles) < n_questions:
        return None

    circles = sorted(circles, key=lambda c: c[1])  # فرز عمودي

    # تجميع في صفوف بفارج ±tolerance
    rows = []
    tol  = max(10, int(np.std([c[1] for c in circles]) * 0.3 + 5))
    current_row = [circles[0]]
    for c in circles[1:]:
        if abs(c[1] - current_row[-1][1]) <= tol:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    # احتفظ فقط بالصفوف التي تحتوي n_options دوائر (أو أقل بقليل)
    valid_rows = [r for r in rows if abs(len(r) - n_options) <= 1]

    if len(valid_rows) < n_questions:
        # محاولة ثانية بتساهل أكبر
        valid_rows = sorted(rows, key=lambda r: abs(len(r) - n_options))[:n_questions]

    valid_rows = valid_rows[:n_questions]
    valid_rows.sort(key=lambda r: np.mean([c[1] for c in r]))   # رتب عمودياً

    grid = []
    for row in valid_rows:
        row_sorted = sorted(row, key=lambda c: c[0], reverse=True)  # RTL: أكبر x أولاً
        # أكمل الصف إذا كان ناقصاً
        while len(row_sorted) < n_options:
            row_sorted.append(None)
        grid.append(row_sorted[:n_options])

    return grid


def read_grid(grid, warped_thresh, radius=11):
    """
    يقرأ الإجابة المُظللة من كل سطر في الشبكة.
    يُرجع قائمة من الإجابات (0-based index أو -1 = فارغ)
    """
    results = []
    H, W    = warped_thresh.shape
    for row in grid:
        counts = []
        for bubble in row:
            if bubble is None:
                counts.append(0)
                continue
            cx, cy = bubble[0], bubble[1]
            x1 = max(0, cx - radius); x2 = min(W, cx + radius)
            y1 = max(0, cy - radius); y2 = min(H, cy + radius)
            counts.append(cv2.countNonZero(warped_thresh[y1:y2, x1:x2]))
        best = int(np.argmax(counts))
        results.append(best if counts[best] > 55 else -1)
    return results


# =====================================================================
# واجهة المستخدم
# =====================================================================
camera_file = st.camera_input("📷 صوِّر ورقة الإجابات — تأكد أن الورقة كاملة داخل الإطار")

if camera_file is not None:
    try:
        pil_img = Image.open(camera_file)
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H_orig, W_orig = img_bgr.shape[:2]

        # ── الخطوة 1: تقويم الورقة ───────────────────────────────────
        paper = find_paper_contour(img_bgr)
        if paper is None:
            st.warning("⚠️ لم أجد حدود الورقة بوضوح — سأعالج الصورة كاملة.")
            warped, W, H = img_bgr.copy(), W_orig, H_orig
        else:
            preview = img_bgr.copy()
            cv2.drawContours(preview, [paper], -1, (0,230,0), 4)
            st.success("✅ تم تحديد حدود الورقة!")
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="الورقة المرصودة")
            warped, W, H = four_point_transform(
                img_bgr, paper.reshape(4,2).astype("float32")
            )

        # ── الخطوة 2: تحسين الصورة ───────────────────────────────────
        warped_gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe        = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        warped_eq    = clahe.apply(warped_gray)
        warped_thresh = cv2.threshold(
            warped_eq, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        # ── الخطوة 3: كشف الدوائر تلقائياً ──────────────────────────
        circles    = detect_bubbles(warped_eq)
        separators = find_section_separators(warped_eq)

        if len(circles) < 15:
            st.error(f"❌ عدد الدوائر المكتشفة قليل جداً ({len(circles)}). تأكد من الإضاءة ووضوح الورقة.")
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="الورقة المُقوَّمة")
            st.stop()

        tf_c, mc_c, match_c, sep_y1, sep_y2 = assign_bubbles_to_sections(
            circles, separators, H
        )

        # ── الخطوة 4: بناء الشبكات وقراءة الإجابات ──────────────────
        tf_grid    = grid_from_circles(tf_c,    n_questions=5,  n_options=2)
        mc_grid    = grid_from_circles(mc_c,    n_questions=10, n_options=4)
        match_grid = grid_from_circles(match_c, n_questions=5,  n_options=5)

        if tf_grid is None or mc_grid is None or match_grid is None:
            st.error("❌ لم أتمكن من تحليل بنية الورقة. جرب تصويرها أكثر وضوحاً مع إضاءة جيدة.")
            with st.expander("تفاصيل للمطور"):
                st.write(f"دوائر TF: {len(tf_c)}, MC: {len(mc_c)}, Match: {len(match_c)}")
                st.write(f"الفواصل: {separators}")
                debug_all = warped.copy()
                for cx,cy,r in circles:
                    cv2.circle(debug_all,(cx,cy),r,(0,255,0),2)
                st.image(cv2.cvtColor(debug_all, cv2.COLOR_BGR2RGB))
            st.stop()

        tf_res    = read_grid(tf_grid,    warped_thresh)
        mc_res    = read_grid(mc_grid,    warped_thresh)
        match_res = read_grid(match_grid, warped_thresh)

        # ── الخطوة 5: التصحيح ────────────────────────────────────────
        score  = 0
        score += sum(1 for a,k in zip(tf_res,    tf_keys)    if a==k)
        score += sum(1 for a,k in zip(mc_res,    mc_keys)    if a==k)
        score += sum(1 for a,k in zip(match_res, match_keys) if a==k)

        # ── عرض النتائج ──────────────────────────────────────────────
        st.divider()
        cs, cp = st.columns(2)
        with cs:
            st.metric("📊 الدرجة الإجمالية", f"{score} / 20")
        with cp:
            pct   = round(score / 20 * 100)
            grade = ("ممتاز 🌟"    if pct>=90 else
                     "جيد جداً ✅" if pct>=75 else
                     "جيد 👍"      if pct>=60 else
                     "مقبول ⚠️"   if pct>=50 else
                     "ضعيف ❌")
            st.metric("النسبة", f"{pct}%  —  {grade}")

        st.divider()

        def lbl(x, kind="mc"):
            if x==-1: return "⬜ فارغ"
            if kind=="tf": return "✔ صح" if x==0 else "✘ خطأ"
            return options_label[x] if x < len(options_label) else "؟"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**صح / خطأ**")
            for i,(a,k) in enumerate(zip(tf_res, tf_keys)):
                st.write(f"{'✅' if a==k else '❌'} س{i+1}: {lbl(a,'tf')} _(صواب: {lbl(k,'tf')})_")
        with c2:
            st.markdown("**الاختياري**")
            for i,(a,k) in enumerate(zip(mc_res, mc_keys)):
                st.write(f"{'✅' if a==k else '❌'} س{i+1}: {lbl(a)} _(صواب: {lbl(k)})_")
        with c3:
            st.markdown("**المزاوجة**")
            for i,(a,k) in enumerate(zip(match_res, match_keys)):
                st.write(f"{'✅' if a==k else '❌'} س{i+1}: {lbl(a)} _(صواب: {lbl(k)})_")

        # ── صور المعالجة ─────────────────────────────────────────────
        st.divider()
        st.subheader("🖼️ مخرجات المعالجة")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="الورقة مُقوَّمة")
        with col2:
            st.image(warped_thresh, caption="خريطة التظليل")

        if debug_mode:
            dbg = warped.copy()
            colors = {0:(0,255,0), 1:(0,165,255), 2:(255,80,80)}
            for sec_idx, (grid, n_opt) in enumerate(
                    [(tf_grid,2),(mc_grid,4),(match_grid,5)]):
                if grid is None: continue
                for q_idx, row in enumerate(grid):
                    for opt_idx, bubble in enumerate(row):
                        if bubble is None: continue
                        cx,cy = bubble[0], bubble[1]
                        cv2.circle(dbg,(cx,cy),13,colors[sec_idx],2)
                        cv2.putText(dbg, str(opt_idx),(cx-5,cy+5),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.38,colors[sec_idx],1)
            # رسم خطوط الفصل
            cv2.line(dbg,(0,sep_y1),(W,sep_y1),(255,255,0),2)
            cv2.line(dbg,(0,sep_y2),(W,sep_y2),(255,255,0),2)
            st.image(cv2.cvtColor(dbg,cv2.COLOR_BGR2RGB),
                     caption="🔬 تشخيص: أخضر=صح/خطأ | برتقالي=اختياري | أحمر=مزاوجة | أصفر=فواصل")
            st.info(f"دوائر مكتشفة: إجمالي={len(circles)} | TF={len(tf_c)} | MC={len(mc_c)} | Match={len(match_c)}\nفواصل y: {sep_y1}, {sep_y2}")

    except Exception as e:
        st.error(f"⚠️ خطأ في المعالجة: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("📷 وجِّه الكاميرا على ورقة الإجابات والتقط صورة لبدء التصحيح.")
    with st.expander("ℹ️ تعليمات"):
        st.markdown("""
**نصائح للحصول على أفضل نتيجة:**
- 💡 إضاءة جيدة وموحدة — تجنب الظل والوهج
- 📄 الورقة مستوية على سطح مستوٍ
- 🔲 خلفية داكنة خلف الورقة
- 📐 الورقة كاملة داخل الإطار بدون قطع

**الجديد:** الكود الآن يكتشف مواضع الدوائر **تلقائياً** من كل صورة
بدلاً من الإحداثيات الثابتة — يعمل مع أي زاوية تصوير.
        """)
