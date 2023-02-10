from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import pickle 
from pydantic import BaseModel, Field

from typing import List
import pandas as pd
import ast
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional

app = FastAPI()


class Movie(BaseModel):
    adult: bool
    belongs_to_collection:  Optional[str]
    budget: int
    categories: str
    homepage: Optional[str]
    id: int
    imdb_id: str
    original_language: str
    original_title: str
    overview: Optional[str]
    popularity: float
    poster_path: Optional[str]
    production_companies: str
    production_countries: str
    release_date: Optional[str]
    revenue: float
    runtime: Optional[float]
    spoken_languages: str
    status: Optional[str]
    tagline: Optional[str]
    title: str
    video: bool 
    vote_average: float
    vote_count: float


class MovieList(BaseModel):
    # OBS: I am new to FastAPI and pydantic 
    # I wanted to pass a list of JSONs or other JSON format
    # but this worked. So for the moment you need to pass:
    # {"items": [record1, record2, etc.]}
    items: List[Movie]
    
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Keep columns of interest
    df = df[['id', 'budget', 'popularity', 'revenue', 'runtime',
                    'vote_average', 'vote_count', 'categories']]

    # Clean the categories column
    df['categories'] = df['categories'].apply(ast.literal_eval)
    df['categories'] = df['categories'].apply(lambda x: [i['name'] for i in x])
    
    # One hot encode categories
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['categories'])
    df = df.join(pd.DataFrame(X, columns=mlb.classes_))
    df = df.drop('categories', axis=1)
    
    # Fill missing values in runtime
    for c in df.columns:
        df[c] = df[c].fillna((df[c].mean()))
        
    # Normalize the data (data is not normally distributed)
    minmaxed_df = preprocessing.MinMaxScaler().fit_transform(df.drop('id',axis=1))
    scaled_df = pd.DataFrame(minmaxed_df, index=df.index, columns=df.columns[1:])
    
    return scaled_df


def fit_model(scaled_df):
    # Now fit for the ideal number of clusters
    kmeans_model = KMeans(n_clusters=7)
    kmeans_model.fit(scaled_df)
    
    return kmeans_model


def predict(kmeans_model, scaled_df):
    # If you pass a sample with 
    categories = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History',
       'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
       'TV Movie', 'Thriller', 'War', 'Western']
    
    for c in categories:
        if c not in scaled_df.columns:
            scaled_df[c] = 0

    # Feature names must be in the same order as they were in fit.
    scaled_df = scaled_df[
        ['budget', 'popularity', 'revenue', 'runtime', 'vote_average',
       'vote_count', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History',
       'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
       'Thriller', 'War', 'Western']
    ]

    scaled_df.to_csv("scaled_df.csv", index=False)
    predictions = list(kmeans_model.predict(scaled_df))
    
    return predictions
    

@app.get("/")
@app.get("/index")
async def root():
    html_content = """
    <html>
        <head>
            <title>Movies Clustering</title>
        </head>
        <body>
            <h1>Use the post methods to train or predict.</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/cluster")
async def scoring_endpoint(data: MovieList):
    data = [record.dict() for record in data.items]
    df = pd.DataFrame(data)
    
    scaled_df = preprocess(df)
    kmeans_model = fit_model(scaled_df)
    
    predictions = kmeans_model.labels_
    df['cluster'] = predictions
    
    return df.to_json(orient="records") 


@app.post("/predict")
async def predict_cluster(data: MovieList):
    data = [record.dict() for record in data.items]
    df = pd.DataFrame(data)

    scaled_df = preprocess(df)

    with open("7-means-clustering.pkl", "rb") as f:
        kmeans_model = pickle.load(f)  
          
    predictions = predict(kmeans_model, scaled_df)
    df['cluster'] = predictions
    
    return df.to_json(orient="records") 
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
