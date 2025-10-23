import pickle
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)
pipe = model['pipeline']
target_names = model['target_names'].tolist()

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")    

app = FastAPI(title='Iris Classifier API',version='1.0')

@app.get('/')
def root():
    return {'status':'ok','usage':'POST /predict'}

@app.post('/predict')
def predict(request: IrisRequest):
    import numpy as np
    X = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])
    y_pred = pipe.predict(X)
    return {'class': target_names[y_pred[0]]}
