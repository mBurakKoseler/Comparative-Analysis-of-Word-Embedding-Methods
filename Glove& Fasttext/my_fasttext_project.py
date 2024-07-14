import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import fasttext
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
df = pd.read_excel('veriSeti.xlsx')

# Orijinal verileri sakla
original_questions = df['Soru'].tolist()
original_answers = df['Cevap'].tolist()
original_categories = df['Kategori'].tolist()

# Gerekli sütunları alalım ve temizleyelim
questions = [preprocess_text(str(q)) for q in original_questions if pd.notnull(q)]
answers = [preprocess_text(str(a)) for a in original_answers if pd.notnull(a)]
categories = [preprocess_text(str(c)) for c in original_categories if pd.notnull(c)]

# Eğitim verilerini dosyaya yaz
with open('questions_answers.txt', 'w', encoding='utf-8') as f:
    for question, answer, category in zip(questions, answers, categories):
        f.write(f"{question} {answer} {category}\n")

# FastText modelini eğit (Skipgram veya CBOW modeli)
model = fasttext.train_unsupervised('questions_answers.txt', model='skipgram', dim=100)
# model = fasttext.train_unsupervised('questions_answers.txt', model='cbow')

# Soruları ve cevapları vektörleştirme
question_vectors = [model.get_sentence_vector(q) for q in questions]
answer_vectors = [model.get_sentence_vector(a) for a in answers]
category_vectors = [model.get_sentence_vector(c) for c in categories]

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
    user_vector = model.get_sentence_vector(user_question)
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

# Stopwords temizleme işleminden sonra kalan kelimeleri yazdırma
# cleaned_questions = [preprocess_text(str(q)) for q in questions]
# cleaned_answers = [preprocess_text(str(a)) for a in answers]
# cleaned_categories = [preprocess_text(str(c)) for c in categories]

# print(f"Cleaned questions: {' '.join(cleaned_questions)}")
# print(f"Cleaned answers: {' '.join(cleaned_answers)}")
# print(f"Cleaned categories: {' '.join(cleaned_categories)}")
