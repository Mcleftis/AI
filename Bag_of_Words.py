import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string


corpus = ["I love NLP", "NLP is fun", "I love machine learning"]


tokens = []
for sentence in corpus:
    words = word_tokenize(sentence.lower())
    words = [w for w in words if w not in stopwords.words('english') and w not in string.punctuation]
    tokens.append(words)

print("Tokens:", tokens)

all_words = [w for sent in tokens for w in sent]
vocab = list(set(all_words))
print("Vocabulary:", vocab)

bow_vectors = []
for sent in tokens:
    bow = Counter(sent)
    vector = [bow.get(word, 0) for word in vocab]
    bow_vectors.append(vector)

print("Bag of Words Vectors:")
for v in bow_vectors:
    print(v)
