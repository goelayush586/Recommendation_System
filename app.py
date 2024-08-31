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

df = pd.read_csv('product.csv', index_col='title')['description']
df = df.dropna()

app=Flask(__name__)



@app.route('/')
def hello():
    return render_template("index.html")



@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit',methods=['POST'])
def submit_data():
    if request.method == 'POST':
        
        # print(request.form['list_jobs'])
       
        text=request.form['list_jobs']
        
        #Define the TFIDF vectorizer that will be used to process the data
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        #Apply this vectorizer to the full dataset to create normalized vectors
        tfidf_matrix = tfidf_vectorizer.fit_transform(df)
        #Get the names of the features
        features = tfidf_vectorizer.get_feature_names_out()
        
        
        def get_closest_neighs(title):
            row = df.index.get_loc(title)
            distances, indices = nbrs.kneighbors(tfidf_matrix.getrow(row))
            return distances,indices
        

        nbrs = NearestNeighbors(n_neighbors=10).fit(tfidf_matrix)
        # unique_org = (df['test'].values)
        distances, indices = get_closest_neighs(text)
        names_similar = pd.Series(indices.flatten()).map(df.reset_index()['title'])
        # result = pd.DataFrame({'distance':distances.flatten(), 'title':names_similar})
        result = pd.DataFrame({'title':names_similar})
        product_names = result['title'].tolist()
        return render_template('index.html', product_names=product_names)  
        
        
        
        
        
        
    #return  'nothing' 
    # return render_template('index.html',tables=[result.to_html(classes='job')],titles=['na','Job'])
        
        
        
        
if __name__ =="__main__":
    
    
    app.run()