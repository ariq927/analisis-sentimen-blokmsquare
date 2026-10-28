
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed"
    / "dataset-blokmsquare_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "hasil"

# Kandidat jumlah klaster
K_RANGE = range(2, 11)

# Nilai awal. Sesuaikan setelah melihat hasil evaluasi.
K_OPTIMAL = 3

TOP_N_WORDS = 10
N_EXAMPLES = 5
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. MEMUAT DATASET
# =========================================================

print("=" * 60)
print("1. MEMUAT DATASET BERSIH")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

if "clean_text" not in df.columns:
    raise ValueError(
        "Kolom 'clean_text' tidak ditemukan dalam dataset."
    )

jumlah_awal = len(df)

# Menghapus teks kosong
df = df.dropna(subset=["clean_text"]).copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""].copy()
df.reset_index(drop=True, inplace=True)

print(f"Jumlah data awal : {jumlah_awal}")
print(f"Jumlah data valid: {len(df)}")

if len(df) < 3:
    raise ValueError("Data terlalu sedikit untuk clustering.")


# =========================================================
# 3. FILTER STOPWORDS DOMAIN
# =========================================================

print("\n" + "=" * 60)
print("2. MENYIAPKAN TEKS")
print("=" * 60)

domain_stopwords = {
    "blok", "square", "mall", "the", "and",
    "place", "for", "you", "here", "with",
    "sini", "buat"
}


def filter_words(text):
    if not isinstance(text, str):
        return ""

    return " ".join(
        word
        for word in text.split()
        if word.lower() not in domain_stopwords
    )


df["text_for_clustering"] = df["clean_text"].apply(filter_words)

# Hapus baris yang menjadi kosong setelah filtering
df = df[df["text_for_clustering"].str.strip() != ""].copy()
df.reset_index(drop=True, inplace=True)

print(f"Jumlah teks setelah filtering: {len(df)}")


# =========================================================
# 4. TF-IDF VECTORIZATION
# =========================================================

print("\n" + "=" * 60)
print("3. EKSTRAKSI FITUR DENGAN TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2)
)

try:
    tfidf_matrix = vectorizer.fit_transform(
        df["text_for_clustering"]
    )
except ValueError as error:
    raise ValueError(
        "TF-IDF gagal dibuat. Periksa teks dan parameter "
        "min_df/max_df. Mungkin terlalu sedikit kata yang valid."
    ) from error

terms = vectorizer.get_feature_names_out()

print(f"Jumlah dokumen : {tfidf_matrix.shape[0]}")
print(f"Jumlah fitur   : {tfidf_matrix.shape[1]}")
print(f"Bentuk matriks : {tfidf_matrix.shape}")

# Simpan daftar fitur
pd.DataFrame({"term": terms}).to_csv(
    OUTPUT_DIR / "tfidf_features.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 5. ELBOW METHOD DAN SILHOUETTE SCORE
# =========================================================

print("\n" + "=" * 60)
print("4. EVALUASI JUMLAH KLASTER")
print("=" * 60)

n_samples = tfidf_matrix.shape[0]

# K harus lebih kecil daripada jumlah dokumen
valid_k = [k for k in K_RANGE if k < n_samples]

if len(valid_k) == 0:
    raise ValueError("Tidak ada kandidat K yang valid.")

inertia_values = []
silhouette_values = []

for k in valid_k:
    print(f"Mengevaluasi K = {k} ...")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = model.fit_predict(tfidf_matrix)

    inertia_values.append(model.inertia_)

    # Gunakan sampel maksimal 1000 ulasan jika data besar
    sample_size = min(1000, n_samples)

    score = silhouette_score(
        tfidf_matrix,
        cluster_labels,
        metric="euclidean",
        sample_size=sample_size if sample_size < n_samples else None,
        random_state=RANDOM_STATE
    )

    silhouette_values.append(score)

    print(
        f"K={k} | Inertia={model.inertia_:.4f} "
        f"| Silhouette={score:.4f}"
    )


# Simpan tabel evaluasi
evaluation_df = pd.DataFrame({
    "K": valid_k,
    "Inertia": inertia_values,
    "Silhouette_Score": silhouette_values
})

evaluation_df.to_csv(
    OUTPUT_DIR / "evaluasi_k.csv",
    index=False,
    encoding="utf-8-sig"
)


# Grafik Elbow Method
plt.figure(figsize=(9, 5))
plt.plot(valid_k, inertia_values, marker="o")
plt.xticks(valid_k)
plt.xlabel("Jumlah Klaster (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Ulasan Blok M Square")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=300)
plt.close()


# Grafik Silhouette Score
plt.figure(figsize=(9, 5))
plt.plot(valid_k, silhouette_values, marker="o")
plt.xticks(valid_k)
plt.xlabel("Jumlah Klaster (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - Ulasan Blok M Square")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "silhouette_score.png", dpi=300)
plt.close()


# Tampilkan rekomendasi berdasarkan Silhouette
best_row = evaluation_df.loc[
    evaluation_df["Silhouette_Score"].idxmax()
]

print("\nRingkasan evaluasi:")
print(evaluation_df.round(4).to_string(index=False))

print(
    f"\nK dengan Silhouette Score tertinggi: "
    f"{int(best_row['K'])}, "
    f"skor={best_row['Silhouette_Score']:.4f}"
)

print("Grafik Elbow dan Silhouette berhasil disimpan.")


# =========================================================
# 6. K-MEANS FINAL
# =========================================================

print("\n" + "=" * 60)
print("5. MENJALANKAN K-MEANS FINAL")
print("=" * 60)

if K_OPTIMAL not in valid_k:
    raise ValueError(
        f"K_OPTIMAL={K_OPTIMAL} tidak termasuk kandidat K valid."
    )

kmeans = KMeans(
    n_clusters=K_OPTIMAL,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(tfidf_matrix)

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("cluster")
    .reset_index(name="jumlah_ulasan")
)

cluster_counts.to_csv(
    OUTPUT_DIR / "jumlah_ulasan_per_cluster.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Jumlah klaster final: {K_OPTIMAL}")
print("\nDistribusi klaster:")
print(cluster_counts.to_string(index=False))


# =========================================================
# 7. KATA KUNCI DAN CONTOH ULASAN
# =========================================================

print("\n" + "=" * 60)
print("6. INTERPRETASI KLASTER")
print("=" * 60)

summary_rows = []

for cluster_id in range(K_OPTIMAL):

    # Ambil indeks dokumen pada klaster terkait
    cluster_indices = df.index[
        df["cluster"] == cluster_id
    ].to_numpy()

    cluster_matrix = tfidf_matrix[cluster_indices]
    centroid = kmeans.cluster_centers_[cluster_id]

    # Kata dengan bobot centroid tertinggi
    top_indices = centroid.argsort()[
        -TOP_N_WORDS:
    ][::-1]

    top_words = [terms[i] for i in top_indices]

    # Pilih contoh ulasan yang dekat dengan centroid
    similarities = cosine_similarity(
        cluster_matrix,
        centroid.reshape(1, -1)
    ).ravel()

    closest_positions = similarities.argsort()[
        ::-1
    ][:N_EXAMPLES]

    examples = [
        df.iloc[cluster_indices[pos]]["clean_text"]
        for pos in closest_positions
    ]

    print("\n" + "-" * 60)
    print(f"KLASTER {cluster_id}")
    print("-" * 60)
    print(f"Jumlah ulasan: {len(cluster_indices)}")
    print(f"Kata kunci: {', '.join(top_words)}")
    print("Contoh ulasan:")

    for example in examples:
        print(f"- {example}")

    summary_rows.append({
        "cluster": cluster_id,
        "jumlah_ulasan": len(cluster_indices),
        "kata_kunci": ", ".join(top_words),
        "contoh_ulasan": " || ".join(examples),
        # Isi setelah membaca kata kunci dan contoh ulasan
        "label_aspek": ""
    })


# Simpan ringkasan untuk pelabelan manual
summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "ringkasan_cluster.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. SIMPAN HASIL CLUSTERING
# =========================================================

result_path = OUTPUT_DIR / "dataset_blokmsquare_clustered.csv"

df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)
print(f"Hasil clustering : {result_path}")
print(f"Folder output    : {OUTPUT_DIR}")
print("File hasil:")
print("- evaluasi_k.csv")
print("- elbow_method.png")
print("- silhouette_score.png")
print("- jumlah_ulasan_per_cluster.csv")
print("- ringkasan_cluster.csv")
print("- tfidf_features.csv")
print("- dataset_blokmsquare_clustered.csv")
