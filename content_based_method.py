
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

# # product_file_import = drive.CreateFile({'id': '1qZD9r6Luv2pOh4jjfOPHIc4blKeoOnn0'})
# # product_file_import.GetContentFile('product.csv')
product = pd.read_csv('product.csv')

product.head()

product.describe(include = 'O')

product['also_bought'] = product['also_bought'].fillna('')
product['also_viewed'] = product['also_viewed'].fillna('')
product['brand'] = product['brand'].fillna('')
product['description'] = product['description'].fillna('')
product['title'] = product['title'].fillna('')

product.shape

product_df = pd.read_csv('product.csv', index_col='title')['description']
print (len(product_df))
product_df

# product_df[title == 'Xtreme Brite Brightening Gel 1oz.']
product_df= product_df[~pd.isnull(product_df)]
print(product_df.shape)
product_df.head()

#Extract text for a particular item
title = 'Xtreme Brite Brightening Gel 1oz.'
text = product_df[title]
#Define the count vectorizer that will be used to process the data
count_vectorizer = CountVectorizer()
#Apply this vectorizer to text to get a sparse matrix of counts
count_matrix = count_vectorizer.fit_transform([text])
#Get the names of the features
features = count_vectorizer.get_feature_names_out()
#Create a series from the sparse matrix
d = pd.Series(count_matrix.toarray().flatten(), 
              index = features).sort_values(ascending=False)

ax = d[:10].plot(kind='bar', figsize=(10,6), width=.8, fontsize=14, rot=45,
            title='Xtreme Brite Brightening Gel Word Counts')
ax.title.set_size(18)

#Define the TFIDF vectorizer that will be used to process the data
tfidf_vectorizer = TfidfVectorizer(analyzer='word',min_df=0,stop_words='english')
#Apply this vectorizer to the full dataset to create normalized vectors
# tfidf_matrix = tfidf_vectorizer.fit_transform(product_df)
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'text_column' is the column containing text data in product_df
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_df)

#Get the names of the features
features = tfidf_vectorizer.get_feature_names_out()
#get the row that contains relevant vector
row = product_df.index.get_loc(title)
#Create a series from the sparse matrix
d = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(), index = features).sort_values(ascending=False)

ax = d[:10].plot(kind='bar', title='Xtreme Brite Brightening Gel TF-IDF Values',
            figsize=(10,6), width=.8, fontsize=14, rot=45 )
ax.title.set_size(20)


# Another example
title1 = 'Stella McCartney Stella'
text = product_df[title1]
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform([text])
features = count_vectorizer.get_feature_names_out()
d = pd.Series(count_matrix.toarray().flatten(), 
              index = features).sort_values(ascending=False)

ax = d[:10].plot(kind='bar', figsize=(8,5), width=.8, fontsize=14, rot=45,
            title='STELLA For Women By STELLA  Word Counts',color = 'C0')
ax.title.set_size(18)

#Define the TFIDF vectorizer that will be used to process the data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#Apply this vectorizer to the full dataset to create normalized vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(product_df)
#Get the names of the features
features = tfidf_vectorizer.get_feature_names_out()
#get the row that contains relevant vector
row = product_df.index.get_loc(title1)
#Create a series from the sparse matrix
d = pd.Series(tfidf_matrix.getrow(row).toarray().flatten(), index = features).sort_values(ascending=False)

ax = d[:10].plot(kind='bar', title='Stella McCartney TF-IDF Values',
            figsize=(10,6), width=.8, fontsize=14, rot=45, color = 'C1' )
ax.title.set_size(20)

nbrs = NearestNeighbors(n_neighbors=10).fit(tfidf_matrix)

def get_closest_neighs(title):
    row = product_df.index.get_loc(title)
    distances, indices = nbrs.kneighbors(tfidf_matrix.getrow(row))
    names_similar = pd.Series(indices.flatten()).map(product_df.reset_index()['title'])
    result = pd.DataFrame({'distance':distances.flatten(), 'title':names_similar})
    return result

get_closest_neighs('Stella McCartney Stella')

get_closest_neighs('Xtreme Brite Brightening Gel 1oz.')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(product['description'])
tfidf_matrix

# how many keys words do we count overall
max(tfidf_matrix[[0]])

cosine_similarity(X = tfidf_matrix, Y=None, dense_output=True)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim[1]

product = product.reset_index()
titles = product['title']
indices = pd.Series(product.index, index=product['title'])

# get title name
indices.head()

def get_highest_cosine_sim(title):
    # get index of a particular item
    idx = indices[title]
    # list score of each title
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sort scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # get 30 highest scores exclude itself
    sim_scores = sim_scores[1:31]
    # print(sim_scores)
    # get item index
    item_indices = [i[0] for i in sim_scores]
    item_distance = [j[1] for j in sim_scores]
    result = pd.DataFrame({'distance':item_distance, 'title': titles.iloc[item_indices]})
    return result

get_highest_cosine_sim('Stella McCartney Stella').head(10)

