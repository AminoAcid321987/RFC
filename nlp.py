import numpy as np

import pandas as pd

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Data_Frame = pd.read_csv("train.txt",delimiter=";", names=["Statement","Emotion"]).head(20)

Var = "Please just write an actual sentence, that's all you have to do."
Var2 = re.sub("[^a-zA-Z]"," ",Var).lower()

MyList = []
for i in Var2.split():
    if i not in stopwords.words("english"):
        MyList.append(i)

NewList = ["running", "cried", "crying", "cries"]
for i in NewList:
    A = WordNetLemmatizer().lemmatize(i)

def Statements(Data_Frame):
    Column_One = Data_Frame
    EmptyList = []
    WordList = []
    for i in Data_Frame:
        New_Sentence = re.sub("[^a-zA-Z]"," ",i).lower()
        for j in New_Sentence.split():
            if j not in stopwords.words("english"):
                K = WordNetLemmatizer().lemmatize(j)
                WordList.append(K)
        L = " ".join(WordList) 
        EmptyList.append(L)  
    return EmptyList

AllStatements = Statements(Data_Frame["Statement"])

from sklearn.feature_extraction.text import CountVectorizer as CV

objct = CV(ngram_range=(1,2))

x = objct.fit_transform(AllStatements)
y = Data_Frame["Emotion"]

from sklearn.ensemble import RandomForestClassifier 

RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(x,y)

Sample_Sentence = Statements(["I am sad."])
objct2 = CV.transform(Sample_Sentence)

objct3 = RFC.predict(objct2)
print(objct3)