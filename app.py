"""
Duygu Analizi Gösterge Paneli (Streamlit)
- Yeni yorum girişi ve anlık tahmin
- Geçmiş tahminlerin görselleştirilmesi
- Model karşılaştırma metrikleri
- Kelime frekans analizi
"""
import os
import sys
import json
from collections import Counter

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess
from database import init_db, insert_prediction, fetch_all_predictions, fetch_stats

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "data")

st.set_page_config(page_title="Duygu Analizi Gösterge Paneli", layout="wide", page_icon="💬")

init_db()


@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "Naive Bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes.joblib")),
        "SVM": joblib.load(os.path.join(MODELS_DIR, "svm.joblib")),
    }
    best_name_raw = joblib.load(os.path.join(MODELS_DIR, "best_model_name.joblib"))
    name_map = {"logistic_regression": "Logistic Regression", "naive_bayes": "Naive Bayes", "svm": "SVM"}
    best_name = name_map.get(best_name_raw, "Logistic Regression")
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    return vectorizer, models, best_name, metrics


@st.cache_data
def load_clean_dataset():
    path = os.path.join(DATA_DIR, "reviews_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


vectorizer, models, best_name, metrics = load_models()

st.title("💬 Duygu Analizi Gösterge Paneli")
st.caption("Amazon ürün yorumları üzerinde eğitilmiş NLP tabanlı duygu sınıflandırma sistemi")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Yorum Analizi", "📊 Model Karşılaştırma", "📈 Geçmiş & İstatistikler", "🔤 Kelime Analizi"
])

# ---------------- TAB 1: Yorum Analizi ----------------
with tab1:
    st.subheader("Yeni Yorum Girin")
    col1, col2 = st.columns([2, 1])
    with col2:
        selected_model_name = st.selectbox("Model seçin", list(models.keys()),
                                            index=list(models.keys()).index(best_name))
        st.info(f"Önerilen en iyi model: **{best_name}**")

    with col1:
        review_input = st.text_area("Yorum metni:", height=120,
                                     placeholder="Örn: This product is amazing, I love it!")

        if st.button("🔍 Analiz Et", type="primary"):
            if review_input.strip():
                clean = preprocess(review_input)
                vec = vectorizer.transform([clean])
                model = models[selected_model_name]
                pred = model.predict(vec)[0]

                # Güven skoru (model destekliyorsa)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec)[0]
                    confidence = float(max(proba))
                else:
                    # LinearSVC için decision_function ile yaklaşık güven
                    score = model.decision_function(vec)[0]
                    confidence = float(1 / (1 + pow(2.71828, -abs(score))))

                sentiment_label = "Pozitif" if pred == 1 else "Negatif"
                emoji = "😊" if pred == 1 else "😞"

                if pred == 1:
                    st.success(f"{emoji} **{sentiment_label}** — Güven Skoru: {confidence:.2%}")
                else:
                    st.error(f"{emoji} **{sentiment_label}** — Güven Skoru: {confidence:.2%}")

                insert_prediction(review_input, sentiment_label, confidence, selected_model_name)
                st.toast("Tahmin veritabanına kaydedildi ✅")
            else:
                st.warning("Lütfen bir yorum metni girin.")

# ---------------- TAB 2: Model Karşılaştırma ----------------
with tab2:
    st.subheader("Model Performans Karşılaştırması")

    metric_rows = []
    for name_key, m in metrics.items():
        display_name = {"logistic_regression": "Logistic Regression",
                         "naive_bayes": "Naive Bayes", "svm": "SVM"}[name_key]
        metric_rows.append({
            "Model": display_name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
        })
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(metrics_df.melt(id_vars="Model", var_name="Metrik", value_name="Değer"),
                 x="Model", y="Değer", color="Metrik", barmode="group",
                 title="Model Metrikleri Karşılaştırması", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix'ler")
    cols = st.columns(3)
    for i, (name_key, m) in enumerate(metrics.items()):
        display_name = {"logistic_regression": "Logistic Regression",
                         "naive_bayes": "Naive Bayes", "svm": "SVM"}[name_key]
        cm = m["confusion_matrix"]
        with cols[i]:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Tahmin", y="Gerçek"),
                                x=["Negatif", "Pozitif"], y=["Negatif", "Pozitif"],
                                title=display_name)
            st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- TAB 3: Geçmiş & İstatistikler ----------------
with tab3:
    st.subheader("Geçmiş Tahminler")
    history_df = fetch_all_predictions()

    if len(history_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Tahmin", len(history_df))
        with col2:
            pos_count = (history_df["predicted_sentiment"] == "Pozitif").sum()
            st.metric("Pozitif Yorumlar", pos_count)
        with col3:
            neg_count = (history_df["predicted_sentiment"] == "Negatif").sum()
            st.metric("Negatif Yorumlar", neg_count)

        fig_pie = px.pie(history_df, names="predicted_sentiment", title="Duygu Dağılımı",
                          color="predicted_sentiment",
                          color_discrete_map={"Pozitif": "#2ecc71", "Negatif": "#e74c3c"})
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("Henüz kayıtlı tahmin yok. 'Yorum Analizi' sekmesinden başlayın.")

# ---------------- TAB 4: Kelime Analizi ----------------
with tab4:
    st.subheader("Eğitim Veri Setinde Kelime Frekans Analizi")
    df_clean = load_clean_dataset()

    if df_clean is not None:
        sentiment_filter = st.radio("Duygu seçin:", ["Tümü", "Pozitif", "Negatif"], horizontal=True)

        if sentiment_filter == "Pozitif":
            subset = df_clean[df_clean["label"] == 1]
        elif sentiment_filter == "Negatif":
            subset = df_clean[df_clean["label"] == 0]
        else:
            subset = df_clean

        all_words = " ".join(subset["clean_text"].dropna().astype(str)).split()
        word_counts = Counter(all_words).most_common(20)
        words_df = pd.DataFrame(word_counts, columns=["Kelime", "Frekans"])

        fig_words = px.bar(words_df, x="Frekans", y="Kelime", orientation="h",
                            title=f"En Sık Kullanılan 20 Kelime ({sentiment_filter})")
        fig_words.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_words, use_container_width=True)

        pos_pct = (df_clean["label"] == 1).mean() * 100
        neg_pct = (df_clean["label"] == 0).mean() * 100
        st.write(f"**Eğitim setinde:** Pozitif %{pos_pct:.1f} | Negatif %{neg_pct:.1f}")
    else:
        st.warning("Temizlenmiş veri seti bulunamadı.")

st.divider()
st.caption("IYD 328 İş Yeri Deneyimi - NLP Duygu Analizi Projesi")
