from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

@app.get("/")
async def recommend(state: str, category: Optional[str] = None):
    df_1 = df[df['state'].str.contains(state, case=False)]
    if category:
        df_2 = df_1[df_1['categories'].str.contains(category, case=False)]
    else:
        df_2 = df_1
    df_3 = df_2.query("avg_rating >= 4")
    return_columns = ['name', 'address', 'state', 'avg_rating', 'categories', 'attributes']
    df_4 = df_3[return_columns]
    features = ['latitude', 'longitude', 'avg_rating', 'review_count']
    X = df_3[features].values
    similarity = cosine_similarity(X)
    index = 0
    sites_index = similarity[index].argsort()[:-11:-1]
    similar_sites = df_4.iloc[sites_index]
    
    return similar_sites.head(10).to_dict()
