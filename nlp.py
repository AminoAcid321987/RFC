import numpy as np

import pandas as pd

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

Data_Frame = pd.read_csv("train.txt",delimiter=";", names=["Statement","Emotion"])

Var = "Please just write an actual sentence, that's all you have to do."
Var2 = re.sub("[^a-zA-Z]"," ",Var).lower()
print(Var2)