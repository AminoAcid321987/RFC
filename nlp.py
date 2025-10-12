import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score as ac

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Data_Frame = pd.read_csv("train.txt",delimiter=";", names=["Statement","Emotion"])

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
    for i in Data_Frame:
        WordList = []
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

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

from sklearn.ensemble import RandomForestClassifier 

RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(x_train,y_train)

Sample_Sentence = Statements(["Today, I am currently furious, this is because I got my exam results back and they were bad."])
objct2 = objct.transform(Sample_Sentence)

objct3 = RFC.predict(x_test)
print(objct3)

ac(objct3,y_test)