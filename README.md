# FastAPI ML Deployment

This project trains a simple unsupervised ML model and uses it to make predictions via API. It uses Sklearn and FastAPI.

## Requirements:

- Python
- [Pipenv](https://pypi.org/project/pipenv/)

## Prepare

- Just run `pipenv install` and you're good to go

## How to Run

- `pipenv run python app.py`
Note: any file change in this folder will trigger a live reload of the server.
- To test your GET/POST methods, go to http://0.0.0.0:8000/docs; use the sample data in 'data'. The names should suggest the use case.

## The data
The dataset is in the file `movies.csv`. It is a subset of 5k movies taken from [kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7?select=movies_metadata.csv). 

## The ML task
The task consists in creating a clustering of those movies based on their categories.

## The API functionality
I built two post methods.
- The '/cluster' endpoint is for clustering a dataset end to end.
- The '/predict' endpoint is for using the '7-means-clustering-pkl' model to predict the cluster of a new data point.

## Project Files
- The 'clustering.ipynb' notebook contains EDA and KMeans clustering implementation. 
- The model is trained, saved and ready to use via 'kmeans-model.pkl'.
In 'app.py', 

## Developer Notes
- I couldn't make the API accept null values in the input (yet) so I just saved non-null dataframe samples for testing the different post methods.
- You can test the functionality using the JSONs in the 'data' folder I created.
- In the 'data' folder, we have different JSON samples to test the API. E.g: 'movies-records.json' is simply a JSON of the initial movies data frame.


