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
df3 = st.cache(pd.read_csv)("MovieRecs2.csv")
dfK = st.cache(pd.read_csv)("MovieRecsKNN.csv")
page_selected = st.sidebar.selectbox("Select Page",("Home","Movie Recommender","Related Graphs","About Us"))

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
    user_input = st.text_input("Enter a movie name....eg: Iron Man","")
    st.write("### Top movies similar to", user_input,":")

    if(user_input != ""):
        try:
            row_data = df3[df3['Movie']==user_input.lower()]
            top10_movie_names = row_data.iloc[0]
            for i in range(1,11):
                movie_name = top10_movie_names[i].replace(" ", "_")
                url = "http://www.omdbapi.com/?s="+movie_name+"&apikey=5c8c455"
                
                data = urllib.request.urlopen(url).read().decode()
                obj = json.loads(data)
                if (obj['Response'] == 'False'):
                    st.write("#### ", top10_movie_names[i])
                    st.write("Movie not found in API")
                else:  
                    id = obj['Search'][0]['imdbID']
                    imdburl= "https://www.imdb.com/title/"+id
                    st.write("#### ", obj['Search'][0]['Title'],': ', imdburl)
                    
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
            

            row_data = dfK[dfK['Movie']==user_input.lower()]
            top10_movie_names = row_data.iloc[0]
            st.write("### Users who liked ", user_input," also liked: ")
            for i in range(1,11):
                movie_name = top10_movie_names[i].replace(" ", "_")
                if movie_name == "0":
                   st.write("#### This movie was not rated by other users in the dataset.")
                   break
                url = "http://www.omdbapi.com/?s="+movie_name+"&apikey=5c8c455"
                
                data = urllib.request.urlopen(url).read().decode()
                obj = json.loads(data)
                if (obj['Response'] == 'False'):
                    st.write("#### ", top10_movie_names[i])
                    st.write("Movie not found in API")
                else:  
                    id = obj['Search'][0]['imdbID']
                    imdburl= "https://www.imdb.com/title/"+id
                    st.write("#### ", obj['Search'][0]['Title'],": ",imdburl)
                    
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

def relatedGraphs():

    graph_selected = st.sidebar.selectbox("Select Graph",("Budget & Revenue","Profitability","Top Actors","Rating vs Popularity","Top Searched Keywords"))

    total_missing = df1.isnull().sum().sort_values(ascending=False)
    percent_missing = ((df1.isnull().sum() / df1.isnull().count()) * 100).sort_values(ascending=False)
    missing_values = pd.concat([total_missing, percent_missing], axis=1, keys=['Total', 'Percent'])

    plt.figure(figsize=(15,10))
    sns.set_style('whitegrid')
    ax = sns.barplot(y=missing_values.index, x=missing_values['Total'], palette="Set2") 
    ax.set_title('Total missing values by attribute')
    ax.set_xlabel('Null count')
    ax.set_ylabel('Attribute Name')

    for p in ax.patches:
        plt.text(p.get_width()+12, p.get_y()+0.65*p.get_height(),
                '{:1.0f}'.format(p.get_width())) 


    if graph_selected == "Budget & Revenue":
        dfBudget = df1[['release_date', 'budget', 'revenue']].dropna()
        dfBudget['release_date'] = pd.DatetimeIndex(dfBudget['release_date']).year
        dfBudget['budget'] = dfBudget['budget'].astype(str).astype(int)
        dfBudget['budget'] = dfBudget['budget'].apply(lambda x: round(x/1000000))
        dfBudget['revenue'] = dfBudget['revenue'].apply(lambda x: round(x/1000000))

        st.write("## Budget & Revenue by Year")
        st.write("Shows scatter plots of budget & revenue by year side by side representing the years from 1880 to 2020")
        plt.figure(figsize=(20,10))
        plt.subplot(1, 2, 1)
        plt.scatter(x = 'release_date' ,y = 'budget' , data = dfBudget)
        plt.xlabel("Year")
        plt.ylabel("Budget (Millions)")
        plt.title("Budget by Year")

        plt.subplot(1, 2, 2)
        plt.scatter(x = 'release_date' ,y = 'revenue' , data = dfBudget)
        plt.xlabel("Year")
        plt.ylabel("Revenue (Millions)")
        plt.title("Revenue by Year")
        plt.show()
        st.pyplot(plt)



        dfBudget = df1[['release_date', 'budget', 'revenue']].dropna()
        dfBudget['release_date'] = pd.DatetimeIndex(dfBudget['release_date']).month
        dfBudget['budget'] = dfBudget['budget'].astype(str).astype(int)
        dfBudget['budget'] = dfBudget['budget'].apply(lambda x: round(x/1000000))
        dfBudget['revenue'] = dfBudget['revenue'].apply(lambda x: round(x/1000000))
        dfBudget = dfBudget.groupby('release_date')['budget', 'revenue'].mean()

        st.write("""## Budget & Average Revenue in Millions per Month""")
        st.write("Shows graph of the budget of movies and their average revenue per month each year")
        plt.figure(figsize=(20,10))
        plt.bar(dfBudget.index , dfBudget['budget'], label='budget', color= 'r')
        plt.bar(dfBudget.index , dfBudget['revenue'], label='revenue', alpha = 0.4, color= 'b')
        plt.xlabel("Month")
        plt.ylabel("Budget / Revenue (Millions)")
        plt.title("Budget/Revenue by Month")
        plt.legend(loc='best')
        plt.show()
        st.pyplot(plt)

    if graph_selected == "Profitability":
        st.title("Movie Profitability")
        st.write("Shows the 20 highest and lowest profitable movies in billions of USD.")
        dfProfit = df1[['original_title', 'budget', 'revenue']].dropna()
        dfProfit['budget'] = dfProfit['budget'].astype(str).astype(int)
        dfProfit['revenue'] = dfProfit['revenue'].astype(int)
        #Remove rows where the budget or revenue is zero because they would not provide a fair comparison
        dfProfit = dfProfit[dfProfit.revenue != 0]
        dfProfit = dfProfit[dfProfit.budget != 0]
        dfProfit['profit'] = dfProfit.apply(lambda row: int(row['revenue']) - row['budget'], axis=1)

        plt.figure(figsize=(20,10))
        sns.set_style('whitegrid')
        plt.subplot(1, 2, 1)
        ax = sns.barplot(y='original_title', x='profit', data=dfProfit.sort_values('profit', ascending=False)[:20], palette="cubehelix") 
        ax.set_title('Top 20 Movies (Best profitability)')
        ax.set_xlabel("Profit (Billion)")
        ax.set_ylabel('Name')

        plt.subplot(1, 2, 2)
        ax = sns.barplot(y='original_title', x='profit', data=dfProfit.sort_values('profit')[:20], palette="cubehelix") 
        ax.set_title('Top 20 Movies (Worst profitability)')
        ax.set_xlabel("Profit (Billion)")
        ax.set_ylabel('Name')
        plt.subplots_adjust(wspace=0.7)
        plt.show()
        st.pyplot(plt)

    if graph_selected == "Top Actors":

        dfBudget = df1[['release_date', 'budget', 'revenue']].dropna()
        dfBudget['release_date'] = pd.DatetimeIndex(dfBudget['release_date']).month
        dfBudget['budget'] = dfBudget['budget'].astype(str).astype(int)
        dfBudget['budget'] = dfBudget['budget'].apply(lambda x: round(x/1000000))
        dfBudget['revenue'] = dfBudget['revenue'].apply(lambda x: round(x/1000000))
        dfBudget = dfBudget.groupby('release_date')['budget', 'revenue'].mean()
        dfYear = df1[['release_date']].dropna()
        dfYear['release_date'] = pd.DatetimeIndex(dfYear['release_date']).year
        dfBudget = dfBudget.groupby('release_date')['budget', 'revenue'].mean()
        dfYear = dfYear.groupby('release_date')['release_date'].count().reset_index(name="count")

        plt.figure(figsize=(20,10))
        sns.set_style('whitegrid')
        ax = sns.lineplot(x='release_date', y='count', data=dfYear) 
        ax.set_title('Movies Per Year')
        ax.set_xlabel("Year")
        ax.set_ylabel('Number of Movies')


        dfCast = df1[['cast']].dropna()
        dfCast = dfCast[dfCast['cast'].map(lambda x: len(x)) > 0]

        dfCast= pd.Series([x for item in dfCast.cast for x in item]).value_counts().reset_index()
        dfCast.columns = ['actor', 'count']
        st.title("Top 20 Actors")
        st.write("Plots the top 20 actors and the number of movies they have had some kind of activity in.")

        plt.figure(figsize=(20,10))
        sns.set_style('whitegrid')
        ax = sns.barplot(y='actor', x='count', data=dfCast.sort_values('count', ascending=False)[:20], palette="cubehelix") 
        ax.set_title('Top 20 Actors')
        ax.set_xlabel("Number of Movies")
        ax.set_ylabel('Actor')

        for p in ax.patches:
            plt.text(p.get_width()+1, p.get_y()+0.65*p.get_height(),
                    '{:1.0f}'.format(p.get_width()))

        st.pyplot(plt)

    if graph_selected == "Rating vs Popularity":
        """# Rating vs Popularity"""
        st.write("Determines if popularity affects whether a rating is high or low.")
        dfPopRating = df1[['popularity', 'vote_average']].dropna()
        dfPopRating['popularity'] = dfPopRating['popularity'].astype(float)
        plt.figure(figsize=(20,10))
        ax = sns.scatterplot(x = 'popularity' ,y = 'vote_average' , data = dfPopRating)
        ax.set_title('Rating vs Popularity')
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Rating")
        st.pyplot(plt)

    if graph_selected == "Top Searched Keywords":
        st.title("Top Searched Keywords")
        st.write("Shows a wordcloud of some top searched keywords when looking for movies.")
        dfOverView = df1[['overview']]
        text = ' '.join(dfOverView['overview'].fillna('').values)
        wordcloud = WordCloud(margin=10, background_color='white', colormap='Blues', width=1200, height=1000).generate(text)
        plt.figure(figsize = (10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Top words in overview', fontsize=20)
        plt.axis('off')
        plt.show()
        st.pyplot(plt)

def selectedPage(page_selected):
    if(page_selected == "Home"):
        homePage()
    elif(page_selected == "About Us"):
        aboutUS()
    elif(page_selected == "Movie Recommender"):
        movieRecommender()
    elif(page_selected == "Related Graphs"):
        relatedGraphs()


selectedPage(page_selected)
#end of main pages
# %%
