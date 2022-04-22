# Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from cltkreaders.lat import LatinTesseraeCorpusReader
from latintools import preprocess

# Define vocabulary
vocabulary = ['pater', 'mater', 'filius', 'filia', 'auunculus', 'patruus']

# Get corpus
T = LatinTesseraeCorpusReader()

# Get texts
docs = [next(T.texts(file, preprocess=preprocess)) for file in T.fileids()]

# Get metadata
dates = [T.metadata('date', file) for file in T.fileids()]
df = pd.DataFrame(dates, columns=['date'], index=T.fileids())
df.to_csv('data/output/dates.tsv', sep='\t')

modes = [T.metadata('mode', file) for file in T.fileids()]
df = pd.DataFrame(modes, columns=['mode'], index=T.fileids())
df.to_csv('data/output/modes.tsv', sep='\t')

# Get approx counts
# TODO: Get true counts

totals = [len(doc.split()) for doc in docs]
df = pd.DataFrame(totals, columns=['total'], index=T.fileids())
df.to_csv('data/output/totals.tsv', sep='\t')

# Vectorize
vectorizer = CountVectorizer(vocabulary=vocabulary)
X = vectorizer.fit_transform(docs)
vocab = vectorizer.get_feature_names_out()
dtm = X.toarray()
df = pd.DataFrame(dtm, columns=vocab, index=T.fileids())
df.to_csv('data/output/counts.tsv', sep='\t')
