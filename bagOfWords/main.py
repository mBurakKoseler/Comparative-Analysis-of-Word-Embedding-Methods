import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Excel dosyasını oku ve metin verilerini düzenle
df = pd.read_excel('veriSeti.xlsx')

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

# Stopwords listesinden çıkar
def remove_stopwords(text, stopwords_list):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(cleaned_words)

for column in ['Soru']:
    df[column] = df[column].apply(lambda x: x.lower())
    df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x))
    df[column] = df[column].apply(remove_stopwords, stopwords_list=user_defined_stopwords)

# Bag of words modelini oluştur
vocabulary = df['Soru'].str.split(expand=True).stack().unique()  # Tüm kelimelerin sırayla alınması için bir vocabulary oluştur
vectorizer = CountVectorizer(vocabulary=vocabulary)
X = vectorizer.fit_transform(df['Soru'])

# Bag of words vektörlerinin uzunluğunu al
print("Bag of Words Vektörlerinin Uzunluğu (Toplam Farklı Kelime Sayısı):", len(vectorizer.get_feature_names_out()))

# Kullanıcıdan cümle al
user_input = input("Bir cümle girin: ")

# Kullanıcının girdisinin bag of words vektörünü oluştur
user_vector = vectorizer.transform([user_input])

# Kullanıcının girdisinin bag of words vektörünü görüntüle
print("Kullanıcının Girdisinin Bag of Words Vektörü:")
print(user_vector.toarray())

# Kullanıcının girdisi ile en yakın soruyu bul
similarities = cosine_similarity(user_vector, X)
most_similar_index = similarities.argmax()

# En yakın sorunun bag of words vektörünü görüntüle
most_similar_question_vector = X[most_similar_index]
print("Girdiye En Benzeyen Sorunun Bag of Words Vektörü:")
print(most_similar_question_vector.toarray())

# En yakın sorunun cevabını yazdır
print("En Yakın Soru:", df['Soru'][most_similar_index])
print("Cevap:", df['Cevap'][most_similar_index])
