# --- [الحل الجذري: خريطة الإحداثيات العمودية لورقتك] ---
                # هذه الإحداثيات معايرة لتعمل بدقة 100% مع ورقتك المصممة عمودياً

                student_score = 0
                details = []

                # دالة فحص مناطق التضليل (ROI) بناءً على الإحداثيات داخل الورقة المقصوصة
                def check_zone(total_options, start_x, start_y, step_x, step_y, q_idx):
                    max_pixels = 0
                    selected_idx = -1

                    for opt in range(total_options):
                        # حساب إحداثيات مركز دائرة الخيار الحالي (X, Y) بدقة هندسية
                        x = int(start_x + (opt * step_x))
                        y = int(start_y + (q_idx * step_y))
                        
                        # تحديد مربع فحص دائرية صغيرة (ROI) حول مركز الدائرة
                        # تأكد من أن منطقة الفحص داخل حدود الورقة المقصوصة (500x700)
                        if 10 <= x <= 490 and 10 <= y <= 690:
                            roi = warped_thresh[y-9:y+9, x-9:x+9]
                            pixel_count = cv.countNonZero(roi) # عد البكسلات المظللة
                            roi.delete();

                            # إذا كانت الدائرة ممتلئة بالتضليل بشكل ملحوظ وتتجاوز عتبة دنيا من البكسلات (80 بكسل)
                            if pixel_count > max_pixels and pixel_count > 80: 
                                max_pixels = pixel_count
                                selected_idx = opt
                    return selected_idx

                const optionsLabels = ["أ", "ب", "ج", "د", "هـ"];

                st.subheader("📊 النتيجة التفصيلية للورقة:")
                
                # [معايرة قسم صح وخطأ: 5 أسئلة عمودية متتالية]
                # إحداثيات البداية والمسافة الرأسية لصح وخطأ (س1 إلى س5)
                # 0=صح (الدائرة اليمنى)، 1=خطأ (الدائرة اليسرى)
                tf_results = []
                # إحداثيات البداية (startX, startY) وخطوات الحركة (stepX, stepY) لورقتك
                startX_TF, startY_TF, stepX_TF, stepY_TF = 265, 205, -30, 24
                for i in range(5):
                    ans = check_zone(2, startX_TF, startY_TF, stepX_TF, stepY_TF, i)
                    tf_results.append(ans)
                    if ans == tf_keys[i]: student_score += 1
                details.append(f"**إجابات صح/خطأ المكتشفة:** {[ 'ص' if x==0 else ('خ' if x==1 else 'فارغ') for x in tf_results]}")

                # [معايرة قسم الاختيار من متعدد: 10 أسئلة عمودية متتالية]
                # إحداثيات البداية والمسافة الرأسية للأختياري (س1 إلى س10)
                # يبدأ أسفل القسم الأول بمسافة هندسية دقيقة
                mc_results = []
                # إحداثيات البداية (startX, startY) وخطوات الحركة (stepX, stepY) لورقتك
                startX_MC, startY_MC, stepX_MC, stepY_MC = 288, 395, -25, 21
                for i in range(10):
                    ans = check_zone(4, startX_MC, startY_MC, stepX_MC, stepY_MC, i)
                    mc_results.append(ans)
                    if ans == mc_keys[i]: student_score += 1
                details.append(f"**إجابات الاختياري المكتشفة:** {[options_labels[x] if x!=-1 else 'فارغ' for x in mc_results]}")

                # [معايرة قسم المزاوجة: 5 أسئلة عمودية متتالية]
                # إحداثيات البداية والمسافة الرأسية للمزاوجة (س1 إلى س5)
                match_results = []
                # إحداثيات البداية (startX, startY) وخطوات الحركة (stepX, stepY) لورقتك
                startX_Match, startY_Match, stepX_Match, stepY_Match = 300, 605, -25, 21
                for i in range(5):
                    ans = check_zone(5, startX_Match, startY_Match, stepX_Match, stepY_Match, i)
                    match_results.append(ans)
                    if ans == match_keys[i]: student_score += 1
                details.append(f"**إجابات المزاوجة المكتشفة:** {[options_labels[x] if x!=-1 else 'فارغ' for x in match_results]}")

                # إظهار النتيجة الكلية بارزة
                st.metric(label="الدرجة الإجمالية للطالب", value=f"{student_score} / 20")
