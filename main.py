from fastapi import FastAPI
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.data.path.append('./nltk_data')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

@app.get("/")
def sentimiento_cercano(state: str = None, categoria: str = None):
    if state is None or categoria is None:
        return 'Gracias por elegir el modelo de recomendación de Datum Tech. Con este servicio, podrás descubrir los mejores lugares para visitar en tu estado, desde restaurantes y discotecas hasta hoteles y más. Solo tienes que ingresar al siguiente enlace: https://api-recomendaciones.onrender.com/docs y empezar a explorar las opciones que más te gusten.'

    df_1 = df[df['state'].str.contains(state, case=False)]
    df_2 = df_1[df_1['categories'].str.contains(categoria, case=False)]
    df_3 = df_2.query("avg_rating >= 4")

    devolver = ['name', 'address', 'avg_rating', 'categories', 'attributes']
    df_4 = df_3[devolver]

    features = ['latitude', 'longitude']
    X = df_3[features].values

    nbrs = NearestNeighbors(n_neighbors=6).fit(X)
    distances, indices = nbrs.kneighbors(X)

    Indice = 0
    Sitios_indice = indices[Indice][1:]
    Sitios_similares = df_4.iloc[Sitios_indice]

    # Para que este ordenado por sitio
    lista_sitios = Sitios_similares.to_dict(orient='records')

    df_5 = pd.DataFrame(lista_sitios)
    df_5['Puntaje de sentimiento'] = df_5['attributes'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df_5['Sentimiento'] = df_5['Puntaje de sentimiento'].apply(lambda x: 'positivo' if x >= 0.0 else 'negativo')
    df_5 = df_5.drop('Puntaje de sentimiento', axis=1)
    return df_5.to_dict(orient='records')
