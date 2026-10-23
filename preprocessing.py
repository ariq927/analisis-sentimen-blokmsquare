import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print("1. Memuat dataset mentah...")
# Baca file dari folder data/raw/
df = pd.read_csv('data/raw/dataset-blokmsquare_raw.csv')

# Pastikan tidak ada data kosong
df = df.dropna(subset=['review_text']).reset_index(drop=True)
print(f"Total ulasan yang akan diproses: {len(df)} ulasan.")

# Inisialisasi Stopwords Bahasa Indonesia
stop_factory = StopWordRemoverFactory()
stopwords_id = set(stop_factory.get_stop_words())

# Tambahkan kata-kata gaul/singkatan ulasan yang tidak bermakna topik
custom_stopwords = {
    'nya', 'yg', 'ga', 'gak', 'nggak', 'bgt', 'banget', 'tp', 'tapi', 
    'kalo', 'kl', 'ke', 'di', 'ini', 'itu', 'udah', 'uda', 'ada', 'sih', 
    'aja', 'saja', 'dgn', 'dengan', 'utk', 'untuk', 'jd', 'jadi', 'deh'
}
all_stopwords = stopwords_id.union(custom_stopwords)

# Kamus normalisasi singkatan umum ulasan
slang_dict = {
    'bgt': 'banget', 'plg': 'paling', 'utk': 'untuk', 'byak': 'banyak',
    'resto': 'restoran', 'worth': 'sepadan', 'hunting': 'mencari'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Case Folding: ubah ke huruf kecil
    text = text.lower()
    
    # 2. Text Cleaning: hapus URL, angka, dan karakter selain huruf a-z
    text = re.sub(r'http\S+|www\S+', '', text)      # hapus link
    text = re.sub(r'\d+', '', text)                  # hapus angka
    text = re.sub(r'[^a-z\s]', ' ', text)           # hanya sisakan huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip()        # rapikan spasi berlebih
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 3. Tokenizing & Normalisasi Singkatan
    words = text.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    
    # 4. Stopword Removal: hapus kata hubung/kata tidak bermakna
    clean_words = [w for w in normalized_words if w not in all_stopwords and len(w) > 2]
    
    return " ".join(clean_words)

print("\n2. Menjalankan Text Cleaning, Case Folding, dan Stopword Removal...")
df['clean_text'] = df['review_text'].apply(clean_text)

# Hapus baris jika ada ulasan yang kosong setelah dibersihkan
df_clean = df[df['clean_text'].str.strip() != ""].copy().reset_index(drop=True)

# 5. Simpan hasil bersih ke folder data/processed/
output_path = 'data/processed/dataset-blokmsquare_clean.csv'
df_clean.to_csv(output_path, index=False)

print(f"\n3. Selesai! File bersih berhasil disimpan ke: {output_path}")
print(f"Total baris bersih yang dihasilkan: {len(df_clean)}")

# Tampilkan preview 5 baris pertama sebelum dan sesudah
print("\n--- Contoh Hasil Preprocessing ---")
for i in range(min(5, len(df_clean))):
    print(f"[{i+1}] ASLI   : {df_clean['review_text'].iloc[i]}")
    print(f"    BERSIH : {df_clean['clean_text'].iloc[i]}\n")