import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. KONFIGURASI
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "data_latih_1500_multibahasa_manado.xlsx"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

NB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "naive_bayes_model.pkl"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ==========================================
# 2. MEMBUAT FOLDER MODEL
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 3. PREPROCESSING TEXT
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Menghapus URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Menghapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Menghapus karakter khusus
    text = re.sub(
        r"[^a-zA-ZÀ-ÿ\s]",
        " ",
        text
    )

    # Menghapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 4. MEMBACA DATASET
# ==========================================

print("=" * 60)
print("MEMBACA DATASET")
print("=" * 60)

df = pd.read_excel(DATA_PATH)

print(f"Jumlah data: {len(df)}")
print()


# ==========================================
# 5. VALIDASI KOLOM
# ==========================================

required_columns = ["text", "label"]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Kolom '{column}' tidak ditemukan dalam dataset."
        )


# ==========================================
# 6. MEMBERSIHKAN DATA
# ==========================================

df = df[["text", "label"]].copy()

df = df.dropna(
    subset=["text", "label"]
)

df["text"] = df["text"].apply(
    clean_text
)

df = df[df["text"].str.strip() != ""]


# ==========================================
# 7. INFORMASI DATA
# ==========================================

print("=" * 60)
print("DISTRIBUSI DATA")
print("=" * 60)

print(
    df["label"].value_counts()
)

print()


# ==========================================
# 8. MEMISAHKAN DATA X DAN Y
# ==========================================

X = df["text"]

y = df["label"]


# ==========================================
# 9. TRAINING DAN TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)

print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")

print()


# ==========================================
# 10. TF-IDF
# ==========================================

print("=" * 60)
print("MEMBUAT TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"Jumlah fitur: {len(vectorizer.get_feature_names_out())}"
)

print()


# ==========================================
# 11. TRAINING NAIVE BAYES
# ==========================================

print("=" * 60)
print("TRAINING NAIVE BAYES")
print("=" * 60)

model = MultinomialNB(
    alpha=1.0
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Training selesai.")

print()


# ==========================================
# 12. PREDIKSI DATA TESTING
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================
# 13. EVALUASI MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print()

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print()


# ==========================================
# 14. MENYIMPAN MODEL
# ==========================================

joblib.dump(
    model,
    NB_MODEL_PATH
)

joblib.dump(
    vectorizer,
    TFIDF_PATH
)


# ==========================================
# 15. INFORMASI PENYIMPANAN
# ==========================================

print("=" * 60)
print("MODEL BERHASIL DISIMPAN")
print("=" * 60)

print(
    f"Model      : {NB_MODEL_PATH}"
)

print(
    f"TF-IDF     : {TFIDF_PATH}"
)

print()

print("Training selesai.")