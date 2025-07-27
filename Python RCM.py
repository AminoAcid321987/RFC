# %%
import pandas as pd

# %%
from sklearn.feature_extraction.text import TfidfVectorizer as Tf

# %%
Data_Frame = pd.read_csv("movies_metadata.csv")
Data_Frame

# %%
Data_Frame
Data_Frame.isnull().sum()

# %%
Threshold = Data_Frame["vote_count"].quantile(0.9)
Threshold

# %%
a = (Data_Frame["vote_count"]/(Data_Frame["vote_count"]+Threshold)) * Data_Frame['vote_average']
b = Threshold/(Threshold+Data_Frame["vote_count"]) * Data_Frame["vote_average"].mean()
Rating = a + b
Rating

# %%
Data_Frame["Rating"] = Rating
Data_Frame

# %%
New_Frame = Data_Frame.sort_values(by="Rating",ascending=False)
New_Frame[["title","Rating","vote_count", "vote_average", "overview"]].head(20)

# %%
New_Frame["overview"].isnull().sum()

# %%
New_Frame["overview"] = New_Frame["overview"].fillna("")
New_Frame["overview"]

# %%
Data_Frame["overview"] = Data_Frame["overview"].fillna("")
Data_Frame["overview"]

# %%
New_Frame["overview"].isnull().sum()

# %%
T = Tf(stop_words="english")
Data_Frame = Data_Frame.head(10)
nl = T.fit_transform(Data_Frame["overview"])

# %%
Series = pd.Series(Data_Frame["title"].index,index=Data_Frame["title"])
Series

# %%
from sklearn.metrics.pairwise import linear_kernel
lk = linear_kernel(nl,nl) 

# %%
def Movie_Rec(movie):
    ID = (Series[movie])
    enu = (list(enumerate(lk[ID])))
    print(sorted(enu,reverse=True))

Movie_Rec("Grumpier Old Men")