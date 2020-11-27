"""
Preprocess the CNN/DM data:
1. Stem
2. Remove the STOPWORDS
3. Select vocabularies
"""

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os.path
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from gensim.models import Phrases
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer


def data_lodaer(input_path, filename):
    with open(f"{input_path}/{filename}", "r") as file:
        lines = [line.rstrip().replace('<S_SEP>', '').replace('-LRB-', '').replace('-RRB-', '') for line in file]
    return lines

# use 'train.article' to build the dict
docs = data_lodaer(input_path=f'./data/cnndm/org_data', filename='train.article')
print(len(docs))

# Tokenize the documents

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

# Lemmatize the documents.
lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

# Compute bigrams.
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
# min_count = 20
# bigram = Phrases(docs, min_count=min_count)
# for idx in range(len(docs)):
#     for token in bigram[docs[idx]]:
#         if '_' in token:
#             # Token is a bigram, add to document.
#             docs[idx].append(token)

# stem the token
ps = PorterStemmer()
docs = [[ps.stem(token) for token in doc] for doc in docs]

# Remove rare and common tokens.
# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 1000 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=1000, no_above=0.5)

# path for saved dictionary: /Users/rachelzheng/opt/anaconda3/lib/python3.7/site-packages/gensim/test/test_data/dict-www-cnndm
# saved /home/rachelzheng/www/venv/lib/python3.6/site-packages/gensim/test/test_data/dict-www-cnndm
dictionary.save(datapath('dict-www-cnndm-unigram'))
# dictionary = Dictionary.load(datapath('dict-www-cnndm'))


# Bag-of-words representation of the documents.
# corpus = [dictionary.doc2bow(doc) for doc in docs]

# Number of unique tokens: 88978 - plus bigram
# Number of unique tokens: 44984 - unigram - 20 documents
# Number of unique tokens: 21185 - unigram - 100 documents
# Number of unique tokens: 6439 - unigram - 1000 docyments
# Number of documents: 287113
print('Number of unique tokens: %d' % len(dictionary))
# print('Number of documents: %d' % len(corpus))

# Make a index to word dictionary.
# temp = dictionary[0]  # This is only to "load" the dictionary.
# id2word = dictionary.id2token
# print(id2word)

# save docs
# for doc in docs:
#     with open('./data/cnndm/corpus/val.txt', 'a') as file:
#         file.write(f"{doc}")
#         file.write("\n")

#output id2word to vocab file
# for i in range(len(id2word)):
#     with open('./data/vocab', 'a') as file:
#         value = id2word[i]
#         file.write(f"{value} {i}")
#         file.write("\n")
