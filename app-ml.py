import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import precision_score
st.title("Machine learning model")
st.subheader("Predict your sentiment :)")
#def local_css(file_name):
 #   with open(file_name) as f:
  #      st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#def remote_css(url):
   # st.markdown(f'<link href="style.css" rel="stylesheet">', unsafe_allow_html=True)    

#local_css("style.css")
#remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

df = pd.read_csv("moviereviews.csv")
df['review'] = df['review'].fillna(' ')
x = df.iloc[:,0].values
df['binary'] = 1
df.loc[df['sentiment']=="negative", 'binary'] = 0
df.loc[df['sentiment']=="positive", 'binary'] = 1
y = df.iloc[:,2].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 0)
Multimodel = Pipeline([('tfidf',TfidfVectorizer(binary = True,max_df=0.611111111111111,norm = 'l2')),("Multinomial",MultinomialNB(alpha = 0.3,class_prior=None, fit_prior=True))])
Multimodel.fit(x_train,y_train)
counts = np.bincount(y_train)
v = np.argmax(counts)
st.subheader("Your Review :  ")
input = st.text_area("\n", "")
print()
predict_me = st.button("Predict_Me")
if predict_me:
  with st.spinner('Wait for it...'):
    time.sleep(5)
 
  y_pred = Multimodel.predict([input])
  if (y_pred == 1):
    st.write('Positive Review with confidence :')
    st.success("Done")
  elif (y_pred == 0):
    st.write('Negative Review with confidence :')
    st.success("Done")
  st.balloons()
#else:
 # st.write('Negative Review')
print()