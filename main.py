from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

class RecommendationRequest(BaseModel):
    state: str
    category: str

@app.post("/recommend")
def recommend_sites(request: RecommendationRequest):
    state = request.state
    category = request.category

    # Filtrar por estado y categoría
    df_1 = df[df['state'].str.contains(state, case=False)]
    df_2 = df_1[df_1['categories'].str.contains(category, case=False)]

    # Filtrar por rating promedio mayor o igual a 4
    df_3 = df_2.query("avg_rating >= 4")

    # Seleccionar columnas deseadas
    columns_to_keep = ['name', 'address', 'state', 'avg_rating', 'categories', 'attributes']
    df_4 = df_3[columns_to_keep]

    # Crear matriz de características
    features = ['latitude', 'longitude', 'avg_rating', 'review_count']
    X = df_3[features].values

    # Calcular similitud coseno
    similarity_matrix = cosine_similarity(X)

    # Recomendar los sitios más similares
    index = 0  # Índice del sitio que se quiere recomendar
    similar_sites_index = similarity_matrix[index].argsort()[:-11:-1]  # Índices de los sitios más similares (excluyendo el sitio original)
    recommended_sites = df_4.iloc[similar_sites_index]  # DataFrame con los sitios más similares

    return recommended_sites.to_dict('records')