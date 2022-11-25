import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
import neattext.functions as nfx
from neattext.functions import clean_text

df = pd.read_csv("labeled_data.csv")
df = df.drop('Unnamed: 0',axis = True)
df = df.drop('count',axis = True)
df = df.drop('offensive_language',axis = True)
df = df.drop('neither',axis = True)
df = df.drop('hate_speech',axis = True)
data = pd.DataFrame()
data['class'] = df['class']
data['tweet'] = df['tweet'].apply(nfx.remove_userhandles)
data['tweet'] = df['tweet'].apply(nfx.remove_puncts)
data['tweet'] = data['tweet'].apply(nfx.remove_stopwords)
data['tweet'] = data['tweet'].apply(nfx.remove_urls,nfx.remove_numbers)
data['tweet'] = data['tweet'].apply(nfx.remove_special_characters)
Xfeatures = data['tweet']
ylabels = data['class']
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=50)
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(pipe_lr, pickle_out)
pickle_out.close()