import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Veri setini oku
df = pd.read_excel('veriSeti.xlsx')

# Orijinal soruları ve cevapları saklamak için yeni sütunlar oluştur
df['Orijinal_Soru'] = df['Soru']
df['Orijinal_Cevap'] = df['Cevap']

# Kullanıcı tarafından belirlenen Türkçe stopwords listesi
user_defined_stopwords = [
    "acaba", "altmış", "altı", "ama", "ancak", "arada", "aslında", "ayrıca", "bana", "bazı", "belki", "ben", "benden",
    "beni", "benim", "beri", "beş", "bile", "bin", "bir", "birçok", "biri", "birkaç", "birkez", "birşey", "birşeyi",
    "biz", "bize", "bizden", "bizi", "bizim", "böyle", "böylece", "bu", "buna", "bunda", "bundan", "bunlar", "bunları",
    "bunların", "bunu", "bunun", "burada", "çok", "çünkü", "da", "daha", "dahi", "de", "defa", "değil", "diğer", "diye",
    "doksan", "dokuz", "dolayı", "dolayısıyla", "dört", "edecek", "eden", "ederek", "edilecek", "ediliyor", "edilmesi",
    "ediyor", "eğer", "elli", "en", "etmesi", "etti", "ettiği", "ettiğini", "gibi", "göre", "halen", "hangi", "hatta",
    "hem", "henüz", "hep", "hepsi", "her", "herhangi", "herkesin", "hiç", "hiçbir", "için", "iki", "ile", "ilgili", "ise",
    "işte", "itibaren", "itibariyle", "kadar", "karşın", "katrilyon", "kendi", "kendilerine", "kendini", "kendisi",
    "kendisine", "kendisini", "kez", "ki", "kim", "kimden", "kime", "kimi", "kimse", "kırk", "milyar", "milyon", "mu",
    "mü", "mı", "ne", "neden", "nedenle", "nerde", "nerede", "nereye", "niye", "niçin", "o", "olan", "olarak",
    "oldu", "olduğu", "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması", "olmayan", "olmaz", "olsa",
    "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan", "onlar", "onlardan", "onları", "onların", "onu",
    "onun", "otuz", "oysa", "öyle", "pek", "rağmen", "sadece", "sanki", "sekiz", "seksen", "sen", "senden", "seni",
    "senin", "siz", "sizden", "sizi", "sizin", "şey", "şeyden", "şeyi", "şeyler", "şöyle", "şu", "şuna", "şunda",
    "şundan", "şunları", "şunu", "tarafından", "trilyon", "tüm", "üç", "üzere", "var", "vardı", "ve", "veya", "ya",
    "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak", "yaptı", "yaptığı", "yaptığını", "yaptıkları", "yedi",
    "yerine", "yetmiş", "yine", "yirmi", "yoksa", "yüz", "zaten"
]

# Stopwords listesinden çıkarma fonksiyonu
def remove_stopwords(text, stopwords_list):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(cleaned_words)

# 'Soru' ve 'Cevap' sütunlarında ön işleme yapma
for column in ['Soru', 'Cevap']:
    df[column] = df[column].apply(lambda x: x.lower())
    df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column] = df[column].apply(lambda x: remove_stopwords(x, user_defined_stopwords))

# 'Soru' ve 'Cevap' sütunlarını birleştir
df['Soru_Cevap'] = df['Soru'] + ' ' + df['Cevap']

# TF-IDF vektörleştirici oluştur ve veriye uygula
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Soru_Cevap'])

# Hedef değişken
y = df['Kategori']

# Veriyi train-test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Modeli değerlendir
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Kullanıcı girdisini al ve ön işleme tabi tut
print("Vektör Boyutu: ", len(vectorizer.vocabulary_))

user_input = input("Bir soru giriniz: ").lower()
user_input = re.sub(r'[^\w\s]', '', user_input)
user_input = remove_stopwords(user_input, user_defined_stopwords)

# Kullanıcı girdisinin TF-IDF vektörünü hesapla
user_input_vector = vectorizer.transform([user_input])

# Kullanıcı girdisi ile veri setindeki sorular arasındaki benzerlikleri hesapla
cosine_similarities = cosine_similarity(user_input_vector, X)

# En benzer soruyu bul
most_similar_index = cosine_similarities.argmax()
most_similar_question = df.iloc[most_similar_index]

print("Kullanıcı Girdisi Vektörü: ", user_input_vector.toarray())
print("En Benzer Soru Vektörü: ", X[most_similar_index].toarray())

# En benzer soru ve cevabını yazdır
print("En benzer soru: ", most_similar_question['Orijinal_Soru'])
print("Cevap: ", most_similar_question['Orijinal_Cevap'])

# Kullanıcı girdisinin ve en benzer sorunun vektörlerini yazdır

