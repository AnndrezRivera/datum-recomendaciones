from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

@app.get("/")
def recomendar_sitios(state: str = None, categoria: str = None):
    if state is None or categoria is None:
        return 'Gracias por elegir el modelo de recomendación de Datum Tech. Con este servicio, podrás descubrir los mejores lugares para visitar en tu estado, desde restaurantes y discotecas hasta hoteles y más. Solo tienes que ingresar al siguiente enlace: https://api-recomendaciones.onrender.com/docs y empezar a explorar las opciones que más te gusten.'

    k = 5
        
    df_1 = df[df['state'].str.contains(state, case=False)]
    df_2 = df_1[df_1['categories'].str.contains(categoria, case=False)]
    df_3 = df_2.query("avg_rating >= 4")

    devolver = ['name', 'address', 'avg_rating', 'categories', 'attributes']
    df_4 = df_3[devolver]

    # Obtener las coordenadas del primer sitio con una puntuación de 5 estrellas
    sitio_referencia = df_3[df_3['avg_rating'] == 5].iloc[0]
    latitud_referencia = sitio_referencia['latitude']
    longitud_referencia = sitio_referencia['longitude']

    # Crear una instancia de NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=k)

    # Ajustar el modelo a los datos de latitud y longitud
    X = df_3[['latitude', 'longitude']].values
    neigh.fit(X)

    # Encontrar los k vecinos más cercanos a un punto en particular
    distances, indices = neigh.kneighbors([[latitud_referencia, longitud_referencia]])

    # Calcular la similitud del coseno
    features = ['latitude', 'longitude', 'avg_rating', 'review_count']
    X = df_3[features].values
    Similitud = cosine_similarity(X)

    # Combinar las puntuaciones de similitud del coseno y kneighbors
    peso_coseno = 0.5
    peso_kneighbors = 0.5
    Similitud_combinada = peso_coseno * Similitud + peso_kneighbors * (1 - distances / distances.max())

    # Obtener los sitios recomendados
    Sitios_indice = Similitud_combinada[Indice].argsort()[:-11:-1]
    Sitios_similares = df_4.iloc[Sitios_indice]

    # Para que este ordenado por sitio
    lista_sitios = Sitios_similares.to_dict(orient='records')

    return lista_sitios[:5]


