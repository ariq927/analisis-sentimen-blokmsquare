"""
preprocessing.py - Preprocessing ulasan Google Maps Blok M Square (Bahasa Indonesia)

Alur per ulasan:
  1. Bersihkan teks (huruf kecil, buang aksen/URL/angka/simbol/emoji)
  2. Deteksi bahasa (buang ulasan non-Indonesia: Inggris, Korea, Jepang, Arab, dll)
  3. Rapikan huruf berulang (seruuu -> seru, mantapp -> mantap)
  4. Normalisasi slang (yg -> yang, gak -> tidak, bgt -> banget, dst)
  5. Buang stopword, TAPI kata negasi (tidak, bukan, belum) dipertahankan
  6. Stemming Sastrawi (pakai cache supaya cepat)
  7. Gabungkan negasi dengan kata sesudahnya (tidak bersih -> tidak_bersih)

Jalankan dari folder proyek:
    python preprocessing.py
"""
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

INPUT_PATH = 'data/raw/dataset-blokmsquare_raw.csv'
OUTPUT_PATH = 'data/processed/dataset-blokmsquare_clean.csv'
DROPPED_PATH = 'data/processed/dataset-blokmsquare_dropped.csv'
TEXT_COL = 'review_text'

USE_STEMMING = True      
MERGE_NEGATION = True    


NEGATIONS = {'tidak', 'bukan', 'belum'}


def _map(target, *sources):
    return {s: target for s in sources}


SLANG = {}
SLANG.update(_map('tidak', 'gak', 'ga', 'gk', 'ngga', 'nggak', 'nggk', 'enggak', 'engga',
                  'ngak', 'tdk', 'tak', 'gag'))
SLANG.update({
    'gaada': 'tidak ada', 'gaenak': 'tidak enak', 'gamau': 'tidak mau', 'gatau': 'tidak tahu',
    'gabisa': 'tidak bisa', 'gabakal': 'tidak bakal', 'gapapa': 'tidak apa',
    'gausa': 'tidak usah', 'gaboleh': 'tidak boleh',
})
SLANG.update(_map('yang', 'yg'))
SLANG.update(_map('dengan', 'dg', 'dgn'))
SLANG.update(_map('dari', 'dr', 'dri'))
SLANG.update(_map('untuk', 'utk', 'untk'))
SLANG.update(_map('karena', 'krn', 'karna', 'krna'))
SLANG.update(_map('jadi', 'jd', 'jdi'))
SLANG.update(_map('juga', 'jg', 'jga'))
SLANG.update(_map('tapi', 'tp', 'tpi'))
SLANG.update(_map('kalau', 'klo', 'kalo', 'kl', 'klu'))
SLANG.update(_map('sama', 'sm', 'sma'))
SLANG.update(_map('saya', 'sy'))
SLANG.update(_map('sudah', 'udh', 'udah', 'uda', 'sdh', 'dah'))
SLANG.update(_map('belum', 'blm', 'blum', 'belom'))
SLANG.update(_map('banget', 'bgt', 'bngt', 'bangt', 'bet'))
SLANG.update(_map('banyak', 'bnyk', 'byk', 'byak', 'bnyak'))
SLANG.update(_map('makan', 'mkn'))
SLANG.update(_map('makanan', 'mknan'))
SLANG.update(_map('langsung', 'lgs', 'lgsg'))
SLANG.update(_map('bosan', 'bosen'))
SLANG.update(_map('kemarin', 'kmrn', 'kmaren', 'kemaren'))
SLANG.update(_map('sekarang', 'skrg', 'skrng', 'sekrang'))
SLANG.update(_map('tempat', 'tmpt'))
SLANG.update(_map('orang', 'org', 'orng'))
SLANG.update(_map('dalam', 'dlm'))
SLANG.update(_map('sebelum', 'sblm'))
SLANG.update(_map('sampai', 'smp', 'sampe'))
SLANG.update(_map('terus', 'trs', 'trus'))
SLANG.update(_map('bisa', 'bs'))
SLANG.update(_map('harus', 'hrs'))
SLANG.update(_map('lagi', 'lg', 'lgi'))
SLANG.update(_map('bukan', 'bkn'))
SLANG.update(_map('emang', 'emg'))
SLANG.update(_map('paling', 'plg'))
SLANG.update(_map('pengen', 'pgn'))
SLANG.update(_map('habis', 'abis'))
SLANG.update(_map('teman', 'tmn', 'tmen', 'temen'))
SLANG.update(_map('jangan', 'jgn'))
SLANG.update(_map('coba', 'cobain', 'nyobain', 'nyoba'))
SLANG.update(_map('mantap', 'mantab', 'mantep', 'mantul', 'mntp'))
SLANG.update(_map('bagus', 'bgus', 'bgs'))
SLANG.update(_map('enak', 'enk', 'enaak'))
SLANG.update(_map('ramai', 'rame', 'rameh', 'ruame', 'rme', 'rmai'))
SLANG.update(_map('murah', 'murmer'))
SLANG.update(_map('keren', 'kece'))
SLANG.update(_map('oke', 'ok', 'okey', 'oks'))
SLANG.update(_map('terimakasih', 'makasih', 'makasi', 'thanks', 'thank', 'tks'))
SLANG.update(_map('rekomendasi', 'recommended', 'recommend', 'recomended', 'recomend',
                  'rekomended', 'rekomen'))
SLANG.update(_map('foto', 'photo'))
SLANG.update(_map('photobooth', 'fotobox', 'photobox', 'fotobooth', 'photoboth'))
SLANG.update(_map('toilet', 'wc'))
SLANG.update(_map('masjid', 'mesjid'))
SLANG.update(_map('restoran', 'resto'))
SLANG.update(_map('sepadan', 'worth'))
SLANG.update(_map('mencari', 'hunting'))
SLANG.update(_map('tukang parkir', 'jukir'))
SLANG.update(_map('rokok', 'ngerokok'))
SLANG.update(_map('satpam', 'security'))
SLANG.update(_map('seperti', 'kayak', 'kyk', 'kek'))
SLANG.update(_map('staf', 'staff'))

EN_TO_ID = {}
EN_TO_ID.update(_map('bagus', 'good', 'nice', 'great', 'amazing', 'awesome', 'perfect', 'excellent'))
EN_TO_ID.update(_map('terbaik', 'best'))
EN_TO_ID.update(_map('keren', 'cool'))
EN_TO_ID.update(_map('seru', 'fun'))
EN_TO_ID.update(_map('suka', 'love'))
EN_TO_ID.update(_map('buruk', 'bad', 'worst'))
EN_TO_ID.update(_map('murah', 'cheap'))
EN_TO_ID.update(_map('bersih', 'clean'))
EN_TO_ID.update(_map('kotor', 'dirty'))
EN_TO_ID.update(_map('padat', 'crowded'))
SLANG.update(EN_TO_ID)

ENGLISH_WORDS = set(EN_TO_ID) | {
    'the', 'and', 'for', 'with', 'this', 'that', 'you', 'are', 'was', 'were', 'is', 'it', 'its',
    'of', 'in', 'on', 'at', 'to', 'from', 'here', 'there', 'very', 'too', 'so', 'but', 'not',
    'can', 'could', 'would', 'have', 'has', 'had', 'lots', 'lot', 'many', 'much', 'more', 'some',
    'any', 'all', 'place', 'food', 'foods', 'prices', 'price', 'shop', 'shops', 'shopping',
    'like', 'really', 'just', 'also', 'when', 'where', 'what', 'which', 'your', 'our', 'they',
    'we', 'my', 'me', 'be', 'been', 'an', 'or', 'if', 'yes', 'visit', 'find', 'one', 'people',
    'street', 'market', 'old', 'big', 'small', 'parking', 'delicious', 'tasty', 'recommended',
    'atmosphere', 'vibes', 'friendly', 'service', 'better', 'overrated', 'pricey', 'look',
    'looking', 'buy', 'eat', 'eating', 'go', 'come', 'get', 'got', 'out', 'around', 'hang',
}

CUSTOM_STOPWORDS = {
    'nya', 'ya', 'yaa', 'yah', 'sih', 'deh', 'dong', 'nih', 'tuh', 'lah', 'loh', 'lho', 'kok',
    'kan', 'mah', 'nah', 'euy', 'woi', 'wah', 'aja', 'ajah', 'doang', 'pun', 'mulu',
    'aku', 'gue', 'gw', 'gua', 'lu', 'lo', 'kamu', 'kita', 'saya', 'kami', 'dia', 'kalian',
    'mereka', 'anda',
    'sini', 'kesini', 'disini', 'situ', 'disitu', 'sana', 'kesana', 'disana',
    'gitu', 'gini', 'begini', 'seperti', 'pas', 'sama', 'buat', 'bikin', 'mau', 'tapi', 'kalau',
    'karena', 'jadi', 'sudah', 'juga', 'dengan', 'untuk', 'yang', 'ini', 'itu', 'ada', 'di', 'ke',
    'dll', 'dsb', 'dkk', 'etc', 'gimana', 'gmn',
}

INTENSIFIERS = {'sangat', 'sekali', 'banget', 'amat', 'terlalu', 'begitu', 'pol'}

DOMAIN_STOPWORDS = {'blok', 'square', 'blokm', 'bms'}

stopwords_id = set(StopWordRemoverFactory().get_stop_words())
ALL_STOPWORDS = (stopwords_id | CUSTOM_STOPWORDS | INTENSIFIERS | DOMAIN_STOPWORDS) - NEGATIONS

PROTECT = {
    'mall', 'full', 'well', 'will', 'still', 'small', 'call', 'hall', 'tall', 'fall', 'all',
    'ill', 'too', 'see', 'bee', 'fee', 'zoo', 'free', 'coffee', 'class', 'boss', 'pass', 'miss',
    'less', 'guess', 'press', 'access', 'business', 'cross', 'stress', 'add', 'odd', 'off',
    'egg', 'feel', 'need', 'keep', 'week', 'look', 'book', 'pool', 'cool', 'door', 'room', 'soon',
}

LAUGH = re.compile(r'^(wk)+k*$|^(ha)+h?$|^(he)+h?$|^(hu)+h?$|^(xi)+x?$|^lol$')

stemmer = StemmerFactory().create_stemmer()


@lru_cache(maxsize=None)
def stem_word(word):
    return stemmer.stem(word)


def is_mostly_latin(text):
    """False kalau teks bukan huruf Latin (Korea, Jepang, Mandarin, Arab, ...)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None  
    latin = sum('LATIN' in unicodedata.name(c, '') for c in letters)
    return latin / len(letters) >= 0.5


def tokenize(text):
    text = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))   
    text = re.sub(r'http\S+|www\.\S+', ' ', text)                      
    text = re.sub(r'\d+', '', text)                                    
    text = re.sub(r'[^a-z\s]', ' ', text)                            
    return text.split()


def normalize_token(word):
    word = re.sub(r'(.)\1{2,}', r'\1', word)                          
    if len(word) >= 3 and word[-1] == word[-2] and word not in PROTECT:
        word = word[:-1]                                              
    return word


def detect_lang(tokens):
    hits = [t for t in tokens if t in ENGLISH_WORDS]
    if len(hits) / len(tokens) < 0.5:
        return 'id'
    if len(tokens) <= 3 and all(t in EN_TO_ID for t in hits):
        return 'id'
    return 'en'


def apply_slang(tokens):
    out = []
    for w in tokens:
        if LAUGH.match(w):
            continue
        out.extend(SLANG.get(w, w).split())
    return out


def keep(word):
    return word in NEGATIONS or (word not in ALL_STOPWORDS and len(word) > 2)


def merge_negation(tokens):
    """['tidak', 'bersih'] -> ['tidak_bersih']; negasi di ujung kalimat dibiarkan."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGATIONS and i + 1 < len(tokens) and tokens[i + 1] not in NEGATIONS:
            out.append(f'{tokens[i]}_{tokens[i + 1]}')
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def clean_tokens(tokens):
    tokens = [w for w in apply_slang(tokens) if keep(w)]
    if USE_STEMMING:
        tokens = [w if (w in NEGATIONS or w in ENGLISH_WORDS) else stem_word(w) for w in tokens]
        tokens = [w for w in apply_slang(tokens) if keep(w)]          
    tokens = [w for i, w in enumerate(tokens) if i == 0 or w != tokens[i - 1]]   
    if MERGE_NEGATION:
        tokens = merge_negation(tokens)
    return tokens


def preprocess(text):
    """Return (lang, clean_text). lang: id / en / other / empty."""
    if not isinstance(text, str) or not text.strip():
        return 'empty', ''
    latin = is_mostly_latin(text)
    if latin is None:
        return 'empty', ''
    if not latin:
        return 'other', ''
    tokens = [normalize_token(w) for w in tokenize(text)]
    if not tokens:
        return 'empty', ''
    lang = detect_lang(tokens)
    if lang != 'id':
        return lang, ''
    return 'id', ' '.join(clean_tokens(tokens))


def main():
    print('1. Memuat dataset mentah...')
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    df = df.drop(columns=['clean_text', 'lang'], errors='ignore')
    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    print(f'   Total ulasan: {len(df)}')

    print('\n2. Preprocessing (stemming pertama kali agak lama, mohon tunggu)...')
    results = [preprocess(t) for t in df[TEXT_COL].astype(str)]
    df['lang'] = [r[0] for r in results]
    df['clean_text'] = [r[1] for r in results]

    is_valid = (df['lang'] == 'id') & (df['clean_text'].str.strip() != '')
    df_clean = df[is_valid].reset_index(drop=True)
    df_drop = df[~is_valid].copy()
    df_drop['alasan'] = df_drop['lang'].map({
        'en': 'bahasa Inggris', 'other': 'bukan huruf Latin', 'empty': 'kosong',
    }).fillna('kosong setelah dibersihkan')

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    df_drop.to_csv(DROPPED_PATH, index=False, encoding='utf-8-sig')

    print(f'\n3. Selesai. Disimpan: {OUTPUT_PATH} ({len(df_clean)} baris)')
    print(f'   Dibuang: {len(df_drop)} baris (lihat {DROPPED_PATH})')
    print(df_drop['alasan'].value_counts().to_string())

    print('\n--- Contoh hasil ---')
    for i in range(min(8, len(df_clean))):
        print(f"[{i + 1}] ASLI   : {df_clean[TEXT_COL].iloc[i]}")
        print(f"    BERSIH : {df_clean['clean_text'].iloc[i]}\n")

    print('--- 40 kata paling sering (cek kata sampah di sini) ---')
    counts = Counter(' '.join(df_clean['clean_text']).split())
    print(', '.join(f'{w}({c})' for w, c in counts.most_common(40)))


if __name__ == '__main__':
    main()