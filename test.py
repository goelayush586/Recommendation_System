from flask import Flask,render_template,redirect,request
import numpy as np
import pandas as pd
import re
from ftfy import fix_text
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

stopw  = set(stopwords.words('english'))

df = pd.read_csv('train.csv', index_col='TITLE')['DESCRIPTION']
# df = pd.read_csv('train.csv')
df = df.dropna()
# print (len(df))
# print (df.head(5))

# df['description'] = df['description'].fillna('')
# df['title'] = df['title'].fillna('')

# title='Xtreme Brite Brightening Gel 1oz.'
#Define the TFIDF vectorizer that will be used to process the data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#Apply this vectorizer to the full dataset to create normalized vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(df)
#Get the names of the features
features = tfidf_vectorizer.get_feature_names()
# #get the row that contains relevant vector
# row = df.index.get_loc(title)
# #Create a series from the sparse matrix
# d = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(), index = features).sort_values(ascending=False)


nbrs = NearestNeighbors(n_neighbors=10).fit(tfidf_matrix)

def get_closest_neighs(title):
    row = df.index.get_loc(title)
    distances, indices = nbrs.kneighbors(tfidf_matrix.getrow(row))
    names_similar = pd.Series(indices.flatten()).map(df.reset_index()['TITLE'])
    result = pd.DataFrame({'distance':distances.flatten(), 'title':names_similar})
    return result

print(get_closest_neighs("adidas Men's Predator 18+ FG Firm Ground Soccer Cleats"))

