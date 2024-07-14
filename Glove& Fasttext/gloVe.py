import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Gerekli kütüphaneleri indir
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    text = text.lower()  # Metni küçük harfe çevir
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopwords kaldır
    text = text.strip()  # Baş ve sondaki boşlukları kaldır
    return text

# GloVe vektörlerini yükleme fonksiyonu
def load_glove_vectors(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        word_vectors = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Cümle vektörlerini hesaplama fonksiyonu
def get_sentence_vector(sentence, word_vectors, vector_size=100):
    words = sentence.split()
    word_vecs = [word_vectors[word] for word in words if word in word_vectors]
    if word_vecs:
        return np.mean(word_vecs, axis=0)
    else:
        return np.zeros(vector_size)

# XLSX dosyasını oku
df = pd.read_excel('C:/Users/brk/Desktop/hocaya atılan/bitirme_study/veriSeti.xlsx')

# Orijinal verileri sakla
original_questions = df['Soru'].tolist()
original_answers = df['Cevap'].tolist()
original_categories = df['Kategori'].tolist()

# Gerekli sütunları alalım ve temizleyelim
questions = [preprocess_text(str(q)) for q in original_questions if pd.notnull(q)]
answers = [preprocess_text(str(a)) for a in original_answers if pd.notnull(a)]
categories = [preprocess_text(str(c)) for c in original_categories if pd.notnull(c)]

# GloVe vektörlerini yükle
word_vectors = load_glove_vectors('C:/Users/brk/Desktop/hocaya atılan/bitirme_study/GloVe/vectors.txt')

# Soruları ve cevapları vektörleştirme
question_vectors = [get_sentence_vector(q, word_vectors) for q in questions]
answer_vectors = [get_sentence_vector(a, word_vectors) for a in answers]
category_vectors = [get_sentence_vector(c, word_vectors) for c in categories]

# En iyi eşleşmeyi bulan fonksiyon
def find_best_match(user_question):
    user_question = preprocess_text(user_question)
    user_vector = get_sentence_vector(user_question, word_vectors)
    similarities = cosine_similarity([user_vector], question_vectors)
    best_match_index = np.argmax(similarities)
    return original_questions[best_match_index], original_answers[best_match_index], original_categories[best_match_index], user_vector

# Kullanıcıdan tekrar tekrar soru al ve en iyi eşleşmeyi bul
while True:
    user_question = input("Bir soru sorun (çıkmak için 'çık' yazın): ")
    if user_question.lower() == 'çık':
        break

    # Kullanıcının girdisinin vektörünü oluşturma ve yazdırma
    similar_question, answer, category, user_vector = find_best_match(user_question)
    print(f"Kullanıcı sorusu vektörü: {user_vector}")

    # En yakın cevabı, kategoriyi ve benzer soruyu bulalım ve yazdıralım
    print(f"Benzer Soru: {similar_question}")
    print(f"Cevap: {answer}\nKategori: {category}")

print("Program sonlandırıldı.")
