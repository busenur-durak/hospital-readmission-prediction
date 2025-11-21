import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------
#                PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="centered"
)

# ---------------------------------------------------
#                CUSTOM CSS (DARK MODERN THEME)
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 40px;
    font-weight: 700;
    color: #FAFAFA;
    text-align: center;
}
.sub-title {
    font-size: 22px;
    color: #CCCCCC;
    text-align: center;
}
.box {
    background-color: #161a23;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}
.footer {
    font-size: 13px;
    text-align: center;
    color: #888888;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
#                SIDEBAR MENU
# ---------------------------------------------------
menu = st.sidebar.radio("Menü", ["🏠 Tahmin", "ℹ️ Hakkında"])

model = joblib.load("models/readmission_xgb.pkl")

# ---------------------------------------------------
#                TAHMIN SAYFASI
# ---------------------------------------------------
if menu == "🏠 Tahmin":

    st.markdown('<p class="big-title">🏥 30 Gün İçinde Tekrar Yatış Tahmini</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Hasta Bilgilerini Girerek Risk Skorunu Hesaplayın</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="box">', unsafe_allow_html=True)

        age = st.selectbox("Yaş Aralığı", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)",
            "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"
        ])

        time = st.slider("Hastanede Kalınan Gün", 1, 14)
        lab = st.slider("Lab Test Sayısı", 0, 100)
        med = st.slider("İlaç Sayısı", 0, 50)
        out = st.slider("Ayaktan Ziyaret", 0, 10)
        emr = st.slider("Acil Servis Ziyareti", 0, 10)
        inp = st.slider("Yatış Sayısı", 0, 10)

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Tahmin Et"):

        sample = {
            "age": age,
            "time_in_hospital": time,
            "num_lab_procedures": lab,
            "num_medications": med,
            "number_outpatient": out,
            "number_emergency": emr,
            "number_inpatient": inp,
        }

        df = pd.DataFrame([sample])

        # Preprocess için kolon doldurma
        expected_cols = model.named_steps["preprocess"].feature_names_in_

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

        score = model.predict_proba(df)[0][1]

        st.success(f"📌 Tahmini Tekrar Yatış Riski: **%{score*100:.2f}**")


# ---------------------------------------------------
#                HAKKINDA / ABOUT SAYFASI
# ---------------------------------------------------
if menu == "ℹ️ Hakkında":

    st.markdown('<p class="big-title">📘 Proje Hakkında</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Proje Amacı
    Bu proje, taburcu edilen hastaların **30 gün içinde tekrar hastaneye yatış yapma olasılığını** tahmin eden 
    bir makine öğrenmesi modelidir.

    ### 🏥 Klinik Önemi
    - Yoğunluk yönetimi  
    - Maliyet azaltma  
    - Hasta güvenliği  
    - Erken uyarı sistemi  

    ### 📊 Kullanılan Yöntemler
    - UCI Diabetic Readmission dataset  
    - Veri temizleme ve eksik değer işlemleri  
    - SMOTE ile dengesiz veri çözümü  
    - XGBoost modelleme  
    - ROC-AUC, PR-AUC değerlendirmeleri  
    - SHAP ile açıklanabilirlik  
    - FastAPI ile API geliştirme  
    - Streamlit ile kullanıcı arayüzü  

    ### 🚀 Sonuç
    Model, riskli hastaların **%85'ini doğru yakalayarak**, 
    doktorlara objektif bir karar destek sistemi sağlar.

    ### 👩‍💻 Geliştiren
    **Busenur Durak**  
    Yönetim Bilişim Sistemleri  
    İzmir Bakırçay Üniversitesi  
    """)

    st.markdown('<p class="footer">© 2025 Hospital Readmission AI — Developed by Busenur Durak</p>', unsafe_allow_html=True)
