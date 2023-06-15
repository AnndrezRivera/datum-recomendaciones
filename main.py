from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

@app.get("/")
def recomendar_sitios(state: str = "CA", categoria: str = "chinese"):
    df_1 = df[df['state'].str.contains(state, case=False)]
    df_2 = df_1[df_1['categories'].str.contains(categoria, case=False)]
    df_3 = df_2.query("avg_rating >= 4")

    devolver = ['name', 'address', 'avg_rating', 'categories', 'attributes']
    df_4 = df_3[devolver]

    features = ['latitude', 'longitude', 'avg_rating', 'review_count']
    X = df_3[features].values

    Similitud = cosine_similarity(X)

    Indice = 0
    Sitios_indice = Similitud[Indice].argsort()[:-11:-1]
    Sitios_similares = df_4.iloc[Sitios_indice]

    # Para que sea una lista
    lista_sitios = Sitios_similares.to_dict(orient='records')

    return lista_sitios[:5]
