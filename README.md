# Comparative-Analysis-of-Word-Embedding-Methods
This project, carried out as part of the KTU CE graduation study, includes how word embedding methods can be used and compared.

    In artificial intelligence, word embedding represents words as numerical vectors, which serves to capture semantic similarities between words. This method improves computational efficiency, reduces memory usage, and enables reuse of pre-trained embeddings in different NLP tasks. Additionally, by understanding the context of words, more accurate and meaningful representation is achieved. These features improve the performance and accuracy of natural language processing applications.

  **The aim of this thesis is to contribute to determining the most suitable methods for specific NLP tasks by comparing the performances of word embedding methods such as BagOfWords, Word2Vec, GloVe, FastText and TF-IDF, which are widely used in the field of natural language processing (NLP).**
In this context, the theoretical foundations and operating principles of each method were examined and performance evaluations were made. While preparing the data set, the question-answer style used by chat bots in the telecom industry was used. The collected data set is not very large, but it contains answers or solutions to frequently asked questions in the telecom industry.

**BagOfWords (BoW)**: BagOfWords is one of the simplest word embedding methods to represent text data. In this method, each word takes a position in a vector and an array is created indicating the presence or absence of each word in the text. This method may lose context information because it does not take word frequencies into account.

**Word2Vec**: Word2Vec is a deep learning model that learns word embedding vectors by taking into account the context around a word. It includes two different models: CBOW (Continuous Bag of Words) and Skip-gram. While the CBOW model predicts the target word using the context around a given word, the Skip-gram model predicts the surrounding words using the context of a word. 

**GloVe (Global Vectors for Word Representation)**:GloVe is a model that learns word embedding vectors using global statistical information. This method is designed to model how often one word co-occurs with another word, based on the co-occurrence of word pairs.

**FastText**: FastText is an extended version of Word2Vec and takes into account sub-words as well as words. Therefore, it can work more effectively with rare or unseen words.

**TF-IDF (Term Frequency-Inverse Document Frequency)**: TF-IDF considers frequency and inverse document frequency to determine the importance of a word in a document. This method is widely used in tasks such as document classification.

The project was prepared together with Ayşegül Akkaya, one of the students of KTU.
**utilized resources**
https://github.com/stanfordnlp/GloVe.git
https://github.com/ahmetax/trstop
The evaluation of word embedding models and deep learning algorithms for Turkish text classification. 4th International Conference on Computer Science and Engineering (2019)
https://builtin.com/machine-learning/pca-in-python
Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language toolkit. O'Reilly Media, Inc
https://medium.com/nerd-for-tech/train-python-code-embedding-with-fasttext-1e225f193cc
https://radimrehurek.com/gensim/models/word2vec.html
https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
