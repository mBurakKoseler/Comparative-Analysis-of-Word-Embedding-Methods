import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# XLSX dosyasını oku
df = pd.read_excel('veriSeti.xlsx', engine='openpyxl')

# Orijinal verileri sakla
original_questions = df['Soru'].tolist()
original_answers = df['Cevap'].tolist()
original_categories = df['Kategori'].tolist()

# Gerekli sütunları alalım ve temizleyelim
questions = [preprocess_text(str(q)) for q in original_questions if pd.notnull(q)]
answers = [preprocess_text(str(a)) for a in original_answers if pd.notnull(a)]
categories = [preprocess_text(str(c)) for c in original_categories if pd.notnull(c)]

# Eğitim verilerini hazırlama
sentences = [q.split() for q in questions] + [a.split() for a in answers] + [c.split() for c in categories]

# Word2Vec modelini eğit
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)  # sg=1 Skip-gram, sg=0 CBOW

# Cümle vektörlerini hesaplama fonksiyonu
def get_sentence_vector(sentence):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Soruları ve cevapları vektörleştirme
question_vectors = [get_sentence_vector(q) for q in questions]
answer_vectors = [get_sentence_vector(a) for a in answers]
category_vectors = [get_sentence_vector(c) for c in categories]

# Vektörlerin uzunluğunu ve farklı kelime sayısını yazdırma
print(f"Question vector length: {len(question_vectors[0])}")
print(f"Answer vector length: {len(answer_vectors[0])}")
print(f"Category vector length: {len(category_vectors[0])}")
print(f"Number of unique words in questions: {len(set(' '.join(questions).split()))}")
print(f"Number of unique words in answers: {len(set(' '.join(answers).split()))}")
print(f"Number of unique words in categories: {len(set(' '.join(categories).split()))}")

# En iyi eşleşmeyi bulan fonksiyon
def find_best_match(user_question):
    user_question = preprocess_text(user_question)
    user_vector = get_sentence_vector(user_question)
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
    print(f"User question vector: {user_vector}")

    # En yakın cevabı, kategoriyi ve benzer soruyu bulalım ve yazdıralım
    print(f"Benzer Soru: {similar_question}")
    print(f"Cevap: {answer}\nKategori: {category}")

print("Program sonlandırıldı.")
