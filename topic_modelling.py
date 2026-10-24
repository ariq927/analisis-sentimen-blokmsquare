import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("1. Memuat dataset bersih...")
df = pd.read_csv('data/processed/dataset-blokmsquare_clean.csv')

domain_stopwords = {'blok', 'square', 'mall', 'the', 'and', 'place', 'for', 'you', 'here', 'with', 'sini', 'buat'}

def filter_words(text):
    if not isinstance(text, str):
        return ""
    return " ".join([w for w in text.split() if w not in domain_stopwords])

df['text_for_clustering'] = df['clean_text'].apply(filter_words)

print("\n2. Melakukan Ekstraksi Fitur dengan TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(df['text_for_clustering'])
print(f"Bentuk matriks TF-IDF: {tfidf_matrix.shape} (dokumen ulasan x kata fitur)")

print("\n3. Mengevaluasi jumlah klaster (Silhouette Score & Elbow Method)...")
k_range = range(2, 7)
inertia_list = []
silhouette_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)
    sil = silhouette_score(tfidf_matrix, labels)
    inertia_list.append(km.inertia_)
    silhouette_list.append(sil)
    print(f"K = {k} | Inertia (Elbow): {km.inertia_:.2f} | Silhouette Score: {sil:.4f}")

K_OPTIMAL = 3
print(f"\n4. Menjalankan K-Means dengan K = {K_OPTIMAL} klaster...")
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print("\n--- Kata Kunci Teratas per Klaster ---")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(K_OPTIMAL):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    total_ulasan = (df['cluster'] == i).sum()
    print(f"\nKlaster {i} (Total: {total_ulasan} ulasan):")
    print(f"Kata kunci: {', '.join(top_terms)}")

label_mapping = {
    0: 'Aspek Suasana & Fasilitas',
    1: 'Aspek Kuliner & Makanan',
    2: 'Aspek Layanan & Tempat'
}
df['aspek_topik'] = df['cluster'].map(label_mapping)

output_path = 'data/processed/dataset_blokmsquare_clustered.csv'
df.to_csv(output_path, index=False)
print(f"\n5. Selesai! Dataset berlabel klaster disimpan ke: {output_path}")