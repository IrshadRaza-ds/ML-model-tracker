from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional
import json

app = FastAPI(title="ML Model Tracker")

DATA_FILE = 'models.json'


# ---------- 1. The order form (Pydantic models) ----------

class MLModel(BaseModel):

    id: Annotated[str, Field(..., description='ID of the model', examples=['M001'])]
    name: Annotated[str, Field(..., description='Name of the model')]
    algorithm: Annotated[
        Literal['logistic_regression', 'random_forest', 'xgboost', 'svm', 'knn', 'neural_network'],
        Field(..., description='Algorithm used')
    ]
    dataset: Annotated[str, Field(..., description='Dataset the model was trained on')]
    accuracy: Annotated[float, Field(..., ge=0, le=1, description='Accuracy between 0 and 1')]
    f1_score: Annotated[float, Field(..., ge=0, le=1, description='F1 score between 0 and 1')]

    # The "smart" part: calculated automatically, never typed by the user
    @computed_field
    @property
    def score(self) -> float:
        return round((self.accuracy + self.f1_score) / 2, 3)

    @computed_field
    @property
    def verdict(self) -> str:
        if self.score >= 0.85:
            return 'Good'
        elif self.score >= 0.70:
            return 'Average'
        else:
            return 'Poor'


class MLModelUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    algorithm: Annotated[
        Optional[Literal['logistic_regression', 'random_forest', 'xgboost', 'svm', 'knn', 'neural_network']],
        Field(default=None)
    ]
    dataset: Annotated[Optional[str], Field(default=None)]
    accuracy: Annotated[Optional[float], Field(default=None, ge=0, le=1)]
    f1_score: Annotated[Optional[float], Field(default=None, ge=0, le=1)]


# ---------- 2. The storage-room helpers ----------

def load_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ---------- 3. The menu (endpoints) ----------

@app.get('/')
def hello():
    return {'message': 'ML Model Tracker API'}


@app.get('/about')
def about():
    return {'message': 'Track your machine learning models and see which ones are good'}


# READ all
@app.get('/view')
def view():
    return load_data()


# READ one
@app.get('/model/{model_id}')
def view_model(model_id: str = Path(..., description='ID of the model', examples=['M001'])):
    data = load_data()
    if model_id in data:
        return data[model_id]
    raise HTTPException(status_code=404, detail='Model not found')


# SORT
@app.get('/sort')
def sort_models(
    sort_by: str = Query(..., description='Sort by accuracy, f1_score or score'),
    order: str = Query('desc', description='asc or desc')
):
    valid_fields = ['accuracy', 'f1_score', 'score']

    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f'Invalid field, select from {valid_fields}')
    if order not in ['asc', 'desc']:
        raise HTTPException(status_code=400, detail='Invalid order, select asc or desc')

    data = load_data()
    return sorted(data.values(), key=lambda x: x.get(sort_by, 0), reverse=(order == 'desc'))


# SEARCH (the extra feature)
@app.get('/search')
def search_models(
    algorithm: Optional[str] = Query(None, description='Filter by algorithm'),
    verdict: Optional[str] = Query(None, description='Filter by Good, Average or Poor')
):
    data = load_data()
    results = list(data.values())

    if algorithm:
        results = [m for m in results if m['algorithm'] == algorithm]
    if verdict:
        results = [m for m in results if m['verdict'].lower() == verdict.lower()]

    return results


# BEST model (another small extra)
@app.get('/best')
def best_model():
    data = load_data()
    if not data:
        raise HTTPException(status_code=404, detail='No models yet')
    return max(data.values(), key=lambda m: m['score'])


# CREATE
@app.post('/create')
def create_model(model: MLModel):
    data = load_data()

    if model.id in data:
        raise HTTPException(status_code=400, detail='Model already exists')

    data[model.id] = model.model_dump(exclude={'id'})
    save_data(data)

    return JSONResponse(status_code=201, content={'message': 'model created successfully'})


# UPDATE
@app.put('/edit/{model_id}')
def update_model(model_id: str, model_update: MLModelUpdate):
    data = load_data()

    if model_id not in data:
        raise HTTPException(status_code=404, detail='Model not found')

    existing_info = data[model_id]
    updated_info = model_update.model_dump(exclude_unset=True)

    for key, value in updated_info.items():
        existing_info[key] = value

    # rebuild the Pydantic object so score + verdict are recalculated
    existing_info['id'] = model_id
    model_obj = MLModel(**existing_info)
    data[model_id] = model_obj.model_dump(exclude={'id'})

    save_data(data)

    return JSONResponse(status_code=200, content={'message': 'model updated'})


# DELETE
@app.delete('/delete/{model_id}')
def delete_model(model_id: str):
    data = load_data()

    if model_id not in data:
        raise HTTPException(status_code=404, detail='Model not found')

    del data[model_id]
    save_data(data)

    return JSONResponse(status_code=200, content={'message': 'model deleted'})