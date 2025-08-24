import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer as Tf

import numpy as np

Data_Frame = pd.read_csv("chicago.txt",sep="\t",header=None,names=["Restaurant_ID","Name","Features"])

Data_Frame
Data_Frame.isnull().sum()

from sklearn.metrics.pairwise import cosine_similarity

Features = Data_Frame["Features"]


Empty_Set = set()

for i in Features:
    Empty_Set.update(i.split())

Dictionary = {}
Enumerated_Value = enumerate(Empty_Set)

for i in Enumerated_Value:
    Dictionary[i[1]]=i[0]

n_restuarants = (len(Features))

n_Features = (len(Empty_Set))

Feature_Matrix = np.zeros((n_restuarants,n_Features),dtype=int)

for i in enumerate(Features):
    for j in (i[1].split()):
        Feature_Matrix[i[0],Dictionary[j]]=1

Trig_Thing = cosine_similarity(Feature_Matrix)
def Restaurant_Rec(Restaurant_ID):
    enu = Trig_Thing[Restaurant_ID]
    Q = (sorted(enu,reverse=True))[1:11]
    print(Q)
    
print(Restaurant_Rec(17))

