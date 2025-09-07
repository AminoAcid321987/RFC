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
    Column_One = Data_Frame["Statement"]
    EmptyList = []
    WordList = []
    for i in Data_Frame["Statement"]:
        New_Sentence = re.sub("[^a-zA-Z]"," ",i).lower()
        for i in New_Sentence.split():
            if i not in stopwords.words("english"):
                K = WordNetLemmatizer().lemmatize(i)
                WordList.append(K)
                Joined = " ".join(WordList)
    print(Joined)

Statements(Data_Frame)




    
