# 📧 Spam Email Detector

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

Aplikasi web untuk mendeteksi apakah sebuah pesan email/SMS termasuk **spam** atau **bukan spam (ham)** menggunakan algoritma Machine Learning. Dibangun sebagai project portofolio untuk membuktikan kemampuan end-to-end ML pipeline.

🔗 **Live Demo:** [spam-detector.streamlit.app](https://spam-detector-s4nd1.streamlit.app/) 

---

## 📸 Screenshot

![App Screenshot](assets/image.png)

---

## 🎯 Tentang Project

Project ini membangun model klasifikasi teks untuk mendeteksi spam menggunakan dataset **SMS Spam Collection** dari UCI Machine Learning Repository. Mencakup seluruh pipeline ML dari eksplorasi data hingga deployment aplikasi web.

**Highlights:**

- Membandingkan **5 algoritma ML** sekaligus untuk mencari model terbaik
- Mencapai akurasi hingga **98%+** pada data testing
- Dilengkapi antarmuka web interaktif yang bisa diakses publik

---

## 📊 Hasil Perbandingan Model

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Naive Bayes         | ~98%     | ~97%      | ~95%     | ~96%     |
| Logistic Regression | ~98%     | ~98%      | ~94%     | ~96%     |
| **SVM**             | **~99%** | **~99%**  | **~97%** | **~98%** |
| Random Forest       | ~97%     | ~100%     | ~88%     | ~93%     |
| KNN                 | ~93%     | ~100%     | ~65%     | ~79%     |

> _Hasil aktual bisa berbeda sedikit tergantung random state. Update tabel ini sesuai hasil notebook kamu._

---

## 🛠️ Tech Stack

| Kategori        | Tools                          |
| --------------- | ------------------------------ |
| Language        | Python 3.12                    |
| ML Framework    | Scikit-Learn                   |
| NLP             | NLTK, TF-IDF Vectorizer        |
| Data Processing | Pandas, NumPy                  |
| Visualization   | Matplotlib, Seaborn, WordCloud |
| Web App         | Streamlit                      |
| Model Saving    | Joblib                         |

---

## 🗂️ Struktur Project

```
spam-detector/
├── app.py                  # Aplikasi Streamlit
├── requirements.txt        # Dependencies
├── model/
│   ├── spam_model.pkl      # Model ML tersimpan
│   └── vectorizer.pkl      # TF-IDF Vectorizer tersimpan
├── notebook/
│   └── spam_detector.ipynb # Notebook eksplorasi & training
├── assets/
│   └── image.png      # Screenshot aplikasi (opsional)
└── README.md
```

---

## ⚙️ Cara Menjalankan Secara Lokal

**1. Clone repository**

```bash
git clone https://github.com/S4nd1Dev/spam-detector.git
cd spam-detector
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**

```bash
streamlit run app.py
```

**4. Buka di browser**

```
http://localhost:8501
```

---

## 🔄 Alur Machine Learning Pipeline

```
Dataset (SMS Spam Collection)
        ↓
   Eksplorasi Data (EDA)
   - Distribusi label
   - Analisis panjang teks
   - Word Cloud
        ↓
   Text Preprocessing
   - Lowercase
   - Hapus karakter spesial
   - Remove stopwords
   - Stemming (PorterStemmer)
        ↓
   Feature Extraction
   - TF-IDF Vectorizer
   - max_features=5000, ngram=(1,2)
        ↓
   Model Training & Comparison
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - KNN
   - SVM
        ↓
   Evaluasi (Accuracy, F1, Confusion Matrix)
        ↓
   Deploy → Streamlit Cloud
```

---

## 📂 Dataset

- **Nama:** SMS Spam Collection Dataset
- **Sumber:** [UCI ML Repository via Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Jumlah data:** 5.572 pesan
- **Distribusi:** ~87% Ham, ~13% Spam

---

## 👤 Author

**Riyan Sandi Prayoga**
Mahasiswa Teknik Informatika — Institut Teknologi Sumatera (ITERA)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rsandip1106-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/rsandip1106)
[![GitHub](https://img.shields.io/badge/GitHub-S4nd1Dev-black?style=flat-square&logo=github)](https://github.com/S4nd1Dev)

---

## 📄 License

Project ini dibuat untuk keperluan pembelajaran dan portofolio.
Dataset bersumber dari UCI Machine Learning Repository (open access).
