import sys
import os

import certifi

from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME,DATA_INGESTION_COLLECTION_NAME
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train",tags=['training'])
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response(content="Training successful!!", media_type="text/plain")
    except Exception as e:
        raise NetworkSecurityException(e, sys)          #type: ignore
    
@app.post("/predict",tags=['prediction'])
async def predict_route(request: Request,file: UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        prepprocessor=load_object("final_models/preprocessor.pkl")
        model=load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=prepprocessor,model=model)
        print(df.iloc[0])
        y_pred=network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'] = df['predicted_column'].map({0: 'Not Phishing', 1: 'Phishing'})
        #return df.to_json()

        df.to_csv("prediction_output/predicted_output.csv")
        table_html = df.to_html(classes="table table-striped")
        #print(table_html)

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)          #type: ignore
    
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)

