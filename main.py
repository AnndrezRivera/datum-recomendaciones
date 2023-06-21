from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



df1 = pd.read_parquet('dataset/archivo1.parquet')
df2 = pd.read_parquet('dataset/archivo2.parquet')
df = pd.concat([df1, df2])

app = FastAPI()

@app.get("/")
def recomendar_sitios(state: str = None, categoria: str = None):
    logger.info(f"Procesando solicitud con state={state} y categoria={categoria}")
    if state is None or categoria is None:
        return 'Gracias por elegir el modelo de recomendación de Datum Tech. Con este servicio, podrás descubrir los mejores lugares para visitar en tu estado, desde restaurantes y discotecas hasta hoteles y más. Solo tienes que ingresar al siguiente enlace: https://api-recomendaciones.onrender.com/docs y empezar a explorar las opciones que más te gusten.'
        
    try:
        df_1 = df[df['state'].str.contains(state, case=False)]
        logger.info(f"df_1.shape={df_1.shape}")
        df_2 = df_1[df_1['categories'].str.contains(categoria, case=False)]
        logger.info(f"df_2.shape={df_2.shape}")
        df_3 = df_2.query("avg_rating >= 4")
        logger.info(f"df_3.shape={df_3.shape}")

        devolver = ['name', 'address', 'avg_rating', 'categories', 'attributes']
        df_4 = df_3[devolver]
        logger.info(f"df_4.shape={df_4.shape}")

        features = ['latitude', 'longitude', 'avg_rating', 'review_count']
        X = df_3[features].values

        if df_4.empty:
            return "Not information found"
       
        Similitud = cosine_similarity(X)

        Indice = 0
        Sitios_indice = Similitud[Indice].argsort()[:-11:-1]
        Sitios_similares = df_4.iloc[Sitios_indice]

        # Para que este ordenado por sitio
        lista_sitios = Sitios_similares.to_dict(orient='records')

        return lista_sitios[:5]
    except Exception as e:
        logger.error(f"Error al procesar la solicitud: {e}")
        return {"error": str(e)}



@app.get("/lugares_cercanos")
def lugares_cercanos(state: str = None, categoria: str = None):
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
    
    if df_5.empty:
        return "Not information found"

    
    return df_5.to_dict(orient='records')


