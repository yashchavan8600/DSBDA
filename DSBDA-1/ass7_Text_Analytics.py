7.1
# Install necessary libraries (if not already installed)
#!pip install nltk

# Import libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.download('all')

# Sample document
document = """
Natural Language Processing (NLP) is a sub-field of artificial intelligence (AI). 
It is focused on enabling computers to understand and process human languages, 
to get valuable insights from them.
"""

print("\nOriginal Document:")
print(document)

# 1. Tokenization Split sentence into words
tokens = word_tokenize(document)
print("\nTokens:")
print(tokens)

# 2. POS Tagging    Identify role of each word
pos_tags = pos_tag(tokens)
print("\nPOS Tags:")
print(pos_tags)

# 3. Stop Words Removal  Remove useless words like:
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nTokens after Stop Words Removal:")
print(filtered_tokens)

# 4. Stemming  Reduce words to root (rough way)
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Tokens:")
print(stemmed_tokens)

# 5. Lemmatization  More accurate than stemming
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nLemmatized Tokens:")
print(lemmatized_tokens)


7.2
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "Natural Language Processing is a part of Artificial Intelligence",
    "Artificial Intelligence and Machine Learning are changing the world",
    "Natural Language is important for communication with machines"
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("\nTF-IDF Representation:\n")
print(df)

df.to_csv("tfidf_output.csv", index=False)

for i in range(len(documents)):
    print(f"\nTop terms in Document {i}:")
    doc_series = df.iloc[i]
    top_terms = doc_series[doc_series > 0].sort_values(ascending=False).head(5)
    print(top_terms)
