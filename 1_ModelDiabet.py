# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:34:21 2021

@author: fadwa
"""

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

st.write(""" 
# Diabetes Detection : Detect if someone has Diabetes """)

page = '''
<style>
body {

  background-color:#fbf4f6;

}
</style>
'''
st.markdown(page, unsafe_allow_html=True)

diab = pd.read_csv("diabetes.csv")


st.sidebar.title("Users informations ")
st.sidebar.header("Choose your information Here")
st.markdown(
    """
<style>
.sidebar .sidebar-content {
   
    background-image: url("https://i.postimg.cc/Kvx1579R/images.jpg");
    background-repeat: no-repeat;
    background-size: 140px , 130px;
    color: black;
    font-family: bold;
    font-size: 43px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: left; color: #e36573;font-size: 27px;'>Data Information :</h1><br>", unsafe_allow_html=True)
#st.subheader("Data Information :")

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: #e36573' if v else '' for v in is_max]
    
dfs = diab.style.apply(highlight_max)
st.dataframe(dfs)

df=diab
df = df.dropna()

st.markdown("<h1 style='text-align: left; color: #e36573;font-size: 27px;'>Data Visualization  :</h1><br>", unsafe_allow_html=True)
pal = {'tested_positive' : 'red', 'tested_negative' : 'orange'}
sns.set(font_scale=1.1) 
g = sns.FacetGrid(df, hue = "class", height = 5 ,palette=pal,hue_kws=dict(marker=["^", "v"]))
g=g.map(sns.scatterplot, "age","plas",s=150, alpha=.7).add_legend()
g.axes[0,0].set_ylabel('glucose').set_color('black')
g.axes[0,0].set_xlabel('Age').set_color('black')

plt.savefig('Plot.png')
st.image('Plot.png')


#determinier les donneér X et target Y (supervisé) en a des données annotées
X = df.drop('class', axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)


#get the features from users 
def get_user_input ():
    
      pregrancies = st.sidebar.slider('pregrancies',0,17,6)
      glucose = st.sidebar.slider('glucose',0,199,148)
      blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
      skin_thickeness = st.sidebar.slider('skin_thickeness',0,99,35)
      insulin = st.sidebar.slider('insulin',0.0,846.0,0.0)
      BMI = st.sidebar.slider('Body mass index',0.0,67.1,33.6)
      pedigree = st.sidebar.slider('pedigree',0.078,2.42,0.627)
      Age = st.sidebar.slider('Age',21,81,50)
      
      
      features = [pregrancies,glucose,blood_pressure,skin_thickeness,insulin,BMI,pedigree,Age]
      return features
      
      
      
#store the user input into variable
st.markdown("<h1 style='text-align: left; color: #e36573;font-size: 27px;'>User Input :</h1><br>", unsafe_allow_html=True)

user_input = get_user_input()
UI = pd.DataFrame([user_input])
UI.columns = ['pregrancies','glucose','blood_pressure','skin_thickeness','insulin','BMI','pedigree','Age']
UIN = UI.append(pd.Series(0, UI.columns), ignore_index=True)
st.dataframe(UIN)

#create machine learning model 
#forest = RandomForestClassifier()
#forest.fit(X_train,y_train)
#y_predict = forest.predict(X_test)

st.markdown("<h1 style='text-align: left; color: #e36573;font-size: 27px;'> Machine Learning Accuracy Score % :</h1>", unsafe_allow_html=True)


svm = SVC()
svm.fit(X_train,y_train)
y_pre = svm.predict(X_test)
score = accuracy_score(y_test, y_pre) * 100
score= format(score, ".0f")
st.subheader(str(accuracy_score(y_test, y_pre))+' '+' ------> '+score+'%')

prediction = svm.predict([user_input])
st.markdown("<h1 style='text-align: left; color: #e36573;font-size: 27px;'> Test Result:</h1>", unsafe_allow_html=True)
#st.subheader('Resultat :')

if prediction[0] == 'tested_positive':
     st.markdown("<h1 style='text-align: left; color: red;font-size: 20px;text-shadow:0 0 60px red;'>Positive  </h1><br>", unsafe_allow_html=True)
elif prediction[0] == 'tested_negative' :
     st.markdown("<h1 style='text-align: left; color: green;font-size: 20px;text-shadow:0 0 60px green;'>Negative </h1><br>", unsafe_allow_html=True)    














