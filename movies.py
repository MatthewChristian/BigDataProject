#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re
from numpy.core.numeric import moveaxis
import streamlit.components.v1 as components
import urllib, json
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.manifold as manifold



from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

#second set of imports
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.manifold as manifold


# from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from pynndescent import NNDescent

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
#end of second set of imports
# In[6]:
#df1 = pd.read_csv("cleaned_df1.csv")
df1 = st.cache(pd.read_csv)("cleaned_df1.csv")
#df2 = st.cache(pd.read_csv)("IMDBMovies.csv")
df3 = st.cache(pd.read_csv)("MovieRecs.csv")
page_selected = st.sidebar.selectbox("Select Page",("Home","Movie Recommender","About Us"))

#start of main pages
def homePage():
    st.write("""# Movie Recommender System
    Welcome to our Movie Recommender System!
    """)
    st.image("https://www.vshsolutions.com/wp-content/uploads/2020/02/recommender-system-for-movie-recommendation.jpg") 
    st.write(df1.head())

def movieRecommender():
    st.title("Movie Recommender")
    """### This recommends movies based on shows of a similar genre."""
    """### Search"""
    user_input = st.text_input("Enter a movie name....eg: Mad Max","")
    st.write("Top movies similar to", user_input,":")

    if(user_input != ""):
        try:
            row_data = df3[df3['Movie']==user_input.lower()]
            top10_movie_names = row_data.iloc[0]
            for i in range(1,11):
                movie_name = top10_movie_names[i].replace(" ", "_")
                url = "http://www.omdbapi.com/?s="+movie_name+"&apikey=5c8c455"
                
                data = urllib.request.urlopen(url).read().decode()
                obj = json.loads(data)
                
                id = obj['Search'][0]['imdbID']
                st.write(obj['Search'][0]['Title'])
                
                url_id = "http://www.omdbapi.com/?i="+id+"&apikey=5c8c455"
                data2 = urllib.request.urlopen(url_id).read().decode()
                obj2 = json.loads(data2)
                
                cols = st.beta_columns(7)
                cols[0].image(obj['Search'][0]['Poster'])
                
                cols[2].write(obj2['Year'])
                cols[3].write(obj2['Rated'])
                cols[4].write(obj2['Genre'])
                cols[5].write(obj2['imdbRating'])
                cols[6].write(obj2['Language'])
                
        except:
            st.write(user_input, " is not an existing movie or movie loaded does not exist in the API.")
        
        
        
def aboutUS():
    st.write("About us")

def selectedPage(page_selected):
    if(page_selected == "Home"):
        homePage()
    elif(page_selected == "About Us"):
        aboutUS()
    elif(page_selected == "Movie Recommender"):
        movieRecommender()


selectedPage(page_selected)
#end of main pages
# %%
