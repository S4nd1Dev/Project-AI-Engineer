import streamlit as st
import numpy as np
import pandas as pd
import zipfile
from urllib.request import urlretrieve
import os
import pickle

# ─── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Sistem Rekomendasi Film",
    page_icon="🎬",
    layout="wide",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = "ml-100k"
MODEL_PATH = "model_embeddings.pkl"
DOT = "dot"
COSINE = "cosine"

GENRE_COLS = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# ─── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📦 Mengunduh dataset MovieLens...")
def load_data():
    if not os.path.exists(DATA_DIR):
        urlretrieve(
            "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "movielens.zip"
        )
        with zipfile.ZipFile("movielens.zip", "r") as z:
            z.extractall()

    users_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
    users = pd.read_csv(
        f"{DATA_DIR}/u.user", sep="|", names=users_cols, encoding="latin-1"
    )

    ratings_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]
    ratings = pd.read_csv(
        f"{DATA_DIR}/u.data", sep="\t", names=ratings_cols, encoding="latin-1"
    )

    movies_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    movies = pd.read_csv(
        f"{DATA_DIR}/u.item", sep="|", names=movies_cols, encoding="latin-1"
    )

    # Preprocessing
    users["user_id"]   = users["user_id"].apply(lambda x: str(x - 1))
    movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
    movies["year"]     = movies["release_date"].apply(lambda x: str(x).split("-")[-1])
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
    ratings["user_id"]  = ratings["user_id"].apply(lambda x: str(x - 1))
    ratings["rating"]   = ratings["rating"].apply(float)

    def get_all_genres(gs):
        active = [genre for genre, g in zip(GENRE_COLS, gs) if g == 1]
        return "-".join(active) if active else "Other"

    movies["all_genres"] = [
        get_all_genres(gs) for gs in zip(*[movies[g] for g in GENRE_COLS])
    ]

    return users, movies, ratings


@st.cache_data(show_spinner="🤖 Melatih model rekomendasi (ini sekali saja)...")
def train_model(_ratings, _movies, _users, embedding_dim=30, init_stddev=0.5, num_iterations=500):
    """Numpy-based matrix factorization (SGD) — no TF needed in Streamlit."""
    n_users  = len(_users)
    n_movies = len(_movies)

    np.random.seed(42)
    U = np.random.normal(0, init_stddev, (n_users, embedding_dim))
    V = np.random.normal(0, init_stddev, (n_movies, embedding_dim))

    lr = 0.01
    ratings_arr = _ratings[["user_id", "movie_id", "rating"]].copy()
    ratings_arr["user_idx"]  = ratings_arr["user_id"].astype(int)
    ratings_arr["movie_idx"] = ratings_arr["movie_id"].astype(int)

    data = ratings_arr[["user_idx", "movie_idx", "rating"]].values

    progress = st.progress(0, text="Training...")
    for it in range(num_iterations):
        np.random.shuffle(data)
        for u_idx, m_idx, r in data:
            u_idx, m_idx = int(u_idx), int(m_idx)
            pred  = U[u_idx].dot(V[m_idx])
            err   = r - pred
            U[u_idx] += lr * err * V[m_idx]
            V[m_idx] += lr * err * U[u_idx]
        if (it + 1) % 50 == 0:
            progress.progress((it + 1) / num_iterations, text=f"Iterasi {it+1}/{num_iterations}")

    progress.empty()
    return {"user_id": U, "movie_id": V}


def compute_scores(query_embedding, item_embeddings, measure=DOT):
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        u = u / (np.linalg.norm(u) + 1e-9)
    return u.dot(V.T)


# ─── App ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 Sistem Rekomendasi Film")
    st.caption("Collaborative Filtering · Matrix Factorization · MovieLens 100K")

    users, movies, ratings = load_data()
    embeddings = train_model(ratings, movies, users)

    st.sidebar.header("⚙️ Pengaturan")
    mode = st.sidebar.radio("Mode Rekomendasi", ["👤 Berdasarkan User", "🎬 Film Serupa"])
    measure = st.sidebar.selectbox("Metode Skor", [COSINE, DOT], format_func=lambda x: x.capitalize())
    k = st.sidebar.slider("Jumlah Rekomendasi", 3, 20, 6)

    st.divider()

    # ── Mode 1: User Recommendations ──────────────────────────────────────────
    if mode == "👤 Berdasarkan User":
        st.subheader("👤 Rekomendasi untuk User")

        max_uid = len(users) - 1
        user_input = st.number_input(
            f"Masukkan User ID (0 – {max_uid})",
            min_value=0, max_value=max_uid, value=0, step=1
        )
        exclude = st.checkbox("Kecualikan film yang sudah dirating", value=True)

        if st.button("🔍 Tampilkan Rekomendasi", use_container_width=True):
            user_idx = int(user_input)
            scores = compute_scores(
                embeddings["user_id"][user_idx],
                embeddings["movie_id"],
                measure
            )
            score_key = f"{measure} score"
            df = pd.DataFrame({
                score_key:   list(scores),
                "movie_id":  movies["movie_id"],
                "Judul":     movies["title"],
                "Genre":     movies["all_genres"],
                "Tahun":     movies["year"],
            })
            if exclude:
                rated = ratings[ratings.user_id == str(user_idx)]["movie_id"].values
                df = df[~df["movie_id"].isin(rated)]

            top = df.sort_values(score_key, ascending=False).head(k).reset_index(drop=True)
            top.index += 1

            st.success(f"✅ Top-{k} rekomendasi untuk User {user_idx}")
            st.dataframe(
                top[["Judul", "Genre", "Tahun", score_key]].rename(
                    columns={score_key: "Skor"}
                ),
                use_container_width=True,
            )

            # Info user
            user_row = users[users.user_id == str(user_idx)]
            if not user_row.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Usia", user_row.iloc[0]["age"])
                col2.metric("Jenis Kelamin", user_row.iloc[0]["sex"])
                col3.metric("Pekerjaan", user_row.iloc[0]["occupation"].title())

    # ── Mode 2: Film Neighbors ─────────────────────────────────────────────────
    else:
        st.subheader("🎬 Film Serupa")

        keyword = st.text_input("Cari judul film (sebagian kata):", placeholder="Contoh: Star Wars")

        if keyword:
            matches = movies[movies["title"].str.contains(keyword, case=False, na=False)]
            if matches.empty:
                st.warning("⚠️ Film tidak ditemukan. Coba kata kunci lain.")
            else:
                selected_title = st.selectbox(
                    "Pilih film:",
                    matches["title"].tolist()
                )

                if st.button("🔍 Cari Film Serupa", use_container_width=True):
                    movie_idx = matches[matches["title"] == selected_title].index[0]
                    scores = compute_scores(
                        embeddings["movie_id"][movie_idx],
                        embeddings["movie_id"],
                        measure
                    )
                    score_key = f"{measure} score"
                    df = pd.DataFrame({
                        score_key: list(scores),
                        "Judul":   movies["title"],
                        "Genre":   movies["all_genres"],
                        "Tahun":   movies["year"],
                    })
                    top = df.sort_values(score_key, ascending=False).head(k + 1)
                    top = top[top["Judul"] != selected_title].head(k).reset_index(drop=True)
                    top.index += 1

                    chosen = movies.iloc[movie_idx]
                    st.info(f"📽️ **{chosen['title']}** ({chosen['year']}) — {chosen['all_genres']}")
                    st.success(f"✅ Top-{k} film serupa")
                    st.dataframe(
                        top[["Judul", "Genre", "Tahun", score_key]].rename(
                            columns={score_key: "Skor"}
                        ),
                        use_container_width=True,
                    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Film", len(movies))
    col2.metric("Total User", len(users))
    col3.metric("Total Rating", len(ratings))


if __name__ == "__main__":
    main()
