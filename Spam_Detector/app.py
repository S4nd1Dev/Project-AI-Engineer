import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Download NLTK resource (otomatis saat pertama kali) ──────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Load model & vectorizer ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load('model/spam_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ── Fungsi preprocessing (HARUS sama persis dengan di notebook) ──────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Spam Detector",
    page_icon  = "📧",
    layout     = "centered"
)

# ── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 700px; }
    .result-spam {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
    .result-ham {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
    .result-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-desc {
        font-size: 14px;
        color: #555;
    }
    .metric-box {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📧 Spam Email Detector")
st.markdown("Masukkan teks email atau pesan, lalu klik **Cek Sekarang** untuk mendeteksi apakah pesan tersebut spam atau bukan.")
st.divider()

# ── Input teks ───────────────────────────────────────────────────────────────
input_text = st.text_area(
    label       = "✏️ Teks Email / Pesan",
    placeholder = "Contoh: Congratulations! You've won a FREE prize worth $1000. Call NOW!",
    height      = 150
)

# Contoh pesan untuk dicoba
with st.expander("💡 Coba contoh pesan"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contoh Spam:**")
        examples_spam = [
            "FREE entry! Win £1000 cash. Text WIN to 12345 now!",
            "URGENT: Your account has been compromised. Click here immediately!",
            "Congratulations! You are selected for a FREE vacation package!",
        ]
        for ex in examples_spam:
            if st.button(f"🔴 {ex[:45]}...", key=ex):
                st.session_state['example'] = ex

    with col2:
        st.markdown("**Contoh Ham:**")
        examples_ham = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Can you send me the notes from today's lecture?",
            "I'll be home late tonight, don't wait for me.",
        ]
        for ex in examples_ham:
            if st.button(f"🟢 {ex[:45]}", key=ex):
                st.session_state['example'] = ex

# Isi text area dengan contoh jika dipilih
if 'example' in st.session_state and not input_text:
    input_text = st.session_state['example']

# ── Tombol prediksi ───────────────────────────────────────────────────────────
col_btn, col_clear = st.columns([3, 1])
with col_btn:
    cek = st.button("🔍 Cek Sekarang", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Hapus", use_container_width=True):
        st.session_state.pop('example', None)
        st.rerun()

# ── Prediksi ─────────────────────────────────────────────────────────────────
if cek:
    if not input_text.strip():
        st.warning("⚠️ Masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Menganalisis pesan..."):
            # Preprocessing & prediksi
            clean      = preprocess_text(input_text)
            vec        = vectorizer.transform([clean])
            prediction = model.predict(vec)[0]
            proba      = model.predict_proba(vec)[0]

            conf_spam = proba[1] * 100
            conf_ham  = proba[0] * 100

        # ── Tampilkan hasil ───────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-spam">
                <div class="result-title">🚨 SPAM Terdeteksi!</div>
                <div class="result-desc">Pesan ini kemungkinan besar adalah spam. Harap berhati-hati.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <div class="result-title">✅ Bukan Spam (Ham)</div>
                <div class="result-desc">Pesan ini terdeteksi sebagai pesan normal yang aman.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence score ──────────────────────────────────────────────
        st.markdown("#### 📊 Confidence Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🚨 Spam", value=f"{conf_spam:.1f}%")
            st.progress(conf_spam / 100)
        with col2:
            st.metric(label="✅ Ham", value=f"{conf_ham:.1f}%")
            st.progress(conf_ham / 100)

        # ── Detail analisis ───────────────────────────────────────────────
        with st.expander("🔬 Lihat Detail Analisis"):
            st.markdown("**Teks asli:**")
            st.info(input_text)
            st.markdown("**Teks setelah preprocessing:**")
            st.code(clean if clean.strip() else "(teks kosong setelah preprocessing)")
            st.markdown("**Jumlah kata setelah preprocessing:**")
            st.write(f"{len(clean.split())} kata")

# ── Divider & info ────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#999; font-size:13px;'>
    Dibuat dengan ❤️ oleh <b>Riyan Sandi Prayoga</b> · Teknik Informatika ITERA<br>
    Model: Machine Learning (Scikit-Learn) · Dataset: SMS Spam Collection (UCI)
</div>
""", unsafe_allow_html=True)
