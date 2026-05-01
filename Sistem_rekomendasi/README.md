# 🎬 Sistem Rekomendasi Film

Aplikasi rekomendasi film berbasis **Collaborative Filtering (Matrix Factorization)** menggunakan dataset [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).

🌐 **Live Demo:** [Klik di sini](#) *(ganti dengan URL Streamlit kamu)*

---

## 📌 Fitur

- **Rekomendasi berdasarkan User** — temukan film yang kemungkinan disukai user tertentu
- **Film Serupa** — temukan film-film mirip berdasarkan embedding
- **Dua metode skor:** Cosine Similarity & Dot Product
- **Filter film yang sudah dirating**

## 🧠 Metode

Model menggunakan **Matrix Factorization** dengan Stochastic Gradient Descent (SGD):

- Setiap user dan film direpresentasikan sebagai vektor embedding berdimensi 30
- Model dilatih untuk meminimalkan error prediksi rating
- Rekomendasi dihasilkan berdasarkan kemiripan embedding

## 🗂️ Struktur Proyek

```
sistem-rekomendasi/
├── app.py                    # Aplikasi Streamlit
├── Sistem_Rekomendasi.ipynb  # Notebook eksperimen asli
├── requirements.txt          # Dependensi Python
├── .gitignore
└── README.md
```

## 🚀 Cara Menjalankan Lokal

```bash
# Clone repo
git clone https://github.com/<username>/sistem-rekomendasi.git
cd sistem-rekomendasi

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo ini → set main file: `app.py`
4. Klik **Deploy!**

> Dataset akan otomatis diunduh saat app pertama kali dijalankan.

## 📊 Dataset

- **MovieLens 100K** — 100.000 rating dari 943 user untuk 1.682 film
- Sumber: [grouplens.org](https://grouplens.org/datasets/movielens/100k/)

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io) — UI
- [NumPy](https://numpy.org) — Matrix Factorization
- [Pandas](https://pandas.pydata.org) — Data processing
