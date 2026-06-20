import pandas as pd
import re 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI

app = FastAPI()

df = pd.read_csv("spam_ham_dataset.csv")

distribusi_nominal = df['label'].value_counts()

with open("stopword.txt", encoding="utf-8") as f:
    stop_words = f.read().split("\n")

kolom_teks = 'text'

def bersihkan_teks_inggris(teks):
    teks = teks.lower()
    teks = re.sub(r'https?://\S+|www\.\S+', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    kata_kata = teks.split()
    kata_bersih = [kata for kata in kata_kata if kata not in stop_words]
    return " ".join(kata_bersih)

df['teks_bersih'] = df[kolom_teks].apply(bersihkan_teks_inggris)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['teks_bersih'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)
prediksi = model.predict(X_test_tfidf)

akurasi = accuracy_score(y_test, prediksi)
presisi = precision_score(y_test, prediksi, pos_label='spam', zero_division=0)
recall = recall_score(y_test, prediksi, pos_label='spam', zero_division=0)
f1 = f1_score(y_test, prediksi, pos_label='spam', zero_division=0)

@app.get("/email")
def cek_email(q: str):
    email = re.search(r"text:\s*'([^']*)'", q)
    out = email.group(1)

    email_bersih = bersihkan_teks_inggris(out)
    email_tfidf = tfidf.transform([email_bersih])
    hasil = model.predict(email_tfidf)[0]

    print(f"Hasil Prediksi : {hasil}")

    if hasil == "spam":
        return "S"
    else:
        return "NS"