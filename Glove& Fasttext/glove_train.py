import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

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
df = pd.read_excel(':/Users/brk/Desktop/hocaya atılan/bitirme_study/veriSeti.xlsx')

# Orijinal verileri sakla
original_questions = df['Soru'].tolist()
original_answers = df['Cevap'].tolist()
original_categories = df['Kategori'].tolist()

# Gerekli sütunları alalım ve temizleyelim
questions = [preprocess_text(str(q)) for q in original_questions if pd.notnull(q)]
answers = [preprocess_text(str(a)) for a in original_answers if pd.notnull(a)]
categories = [preprocess_text(str(c)) for c in original_categories if pd.notnull(c)]

# Eğitim verilerini dosyaya yaz
with open('glove_train.txt', 'w', encoding='utf-8') as f:
    for sentence in questions + answers + categories:
        f.write(sentence + '\n')

