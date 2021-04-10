#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


#df1 = pd.read_csv("cleaned_df1.csv")
df1 = st.cache(pd.read_csv)("cleaned_df1.csv")


# In[7]:


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


# In[18]:


dfBudget = df1[['release_date', 'budget', 'revenue']].dropna()
dfBudget['release_date'] = pd.DatetimeIndex(dfBudget['release_date']).year
dfBudget['budget'] = dfBudget['budget'].astype(str).astype(int)
dfBudget['budget'] = dfBudget['budget'].apply(lambda x: round(x/1000000))
dfBudget['revenue'] = dfBudget['revenue'].apply(lambda x: round(x/1000000))


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


# In[19]:


dfBudget = df1[['release_date', 'budget', 'revenue']].dropna()
dfBudget['release_date'] = pd.DatetimeIndex(dfBudget['release_date']).month
dfBudget['budget'] = dfBudget['budget'].astype(str).astype(int)
dfBudget['budget'] = dfBudget['budget'].apply(lambda x: round(x/1000000))
dfBudget['revenue'] = dfBudget['revenue'].apply(lambda x: round(x/1000000))
dfBudget = dfBudget.groupby('release_date')['budget', 'revenue'].mean()

plt.figure(figsize=(20,10))
plt.bar(dfBudget.index , dfBudget['budget'], label='budget', color= 'r')
plt.bar(dfBudget.index , dfBudget['revenue'], label='revenue', alpha = 0.4, color= 'b')
plt.xlabel("Month")
plt.ylabel("Budget / Revenue (Millions)")
plt.title("Budget/Revenue by Month")
plt.legend(loc='best')
plt.show()
st.pyplot(plt)


# In[20]:


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


# In[11]:


dfYear = df1[['release_date']].dropna()
dfYear['release_date'] = pd.DatetimeIndex(dfYear['release_date']).year
dfdfBudget = dfBudget.groupby('release_date')['budget', 'revenue'].mean()
dfYear = dfYear.groupby('release_date')['release_date'].count().reset_index(name="count")

plt.figure(figsize=(20,10))
sns.set_style('whitegrid')
ax = sns.lineplot(x='release_date', y='count', data=dfYear) 
ax.set_title('Movies Per Year')
ax.set_xlabel("Year")
ax.set_ylabel('Number of Movies')


# In[21]:


dfCast = df1[['cast']].dropna()
dfCast = dfCast[dfCast['cast'].map(lambda x: len(x)) > 0]

dfCast= pd.Series([x for item in dfCast.cast for x in item]).value_counts().reset_index()
dfCast.columns = ['actor', 'count']

plt.figure(figsize=(20,10))
sns.set_style('whitegrid')
ax = sns.barplot(y='actor', x='count', data=dfCast.sort_values('count', ascending=False)[:20], palette="crest") 
ax.set_title('Top 20 Actors')
ax.set_xlabel("Number of Movies")
ax.set_ylabel('Actor')

for p in ax.patches:
    plt.text(p.get_width()+1, p.get_y()+0.65*p.get_height(),
             '{:1.0f}'.format(p.get_width()))

st.pyplot(plt)


# In[22]:


dfPopRating = df1[['popularity', 'vote_average']].dropna()
dfPopRating['popularity'] = dfPopRating['popularity'].astype(float)
plt.figure(figsize=(20,10))
ax = sns.scatterplot(x = 'popularity' ,y = 'vote_average' , data = dfPopRating)
ax.set_title('Rating vs Popularity')
ax.set_xlabel("Popularity")
ax.set_ylabel("Rating")
st.pyplot(plt)


# In[23]:


dfOverView = df1[['overview']]
text = ' '.join(dfOverView['overview'].fillna('').values)
wordcloud = WordCloud(margin=10, background_color='white', colormap='Blues', width=1200, height=1000).generate(text)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Top words in overview', fontsize=20)
plt.axis('off')
plt.show()
st.pyplot(plt)


# In[ ]:




