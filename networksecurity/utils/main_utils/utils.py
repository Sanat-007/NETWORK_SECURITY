import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as yaml_file:
            yaml.dump(content,yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def save_numpy_array_data(file_path:str,array:np.array):        #type: ignore
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)

    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def save_object(file_path:str,obj:object):
    try:
        logging.info("Entered the save_object method of Main Utils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited the save_object method of Main Utils class")
    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def load_object(file_path:str)->object:
    try:
        logging.info("Entered the load_object method of Main Utils class")
        if not os.path.exists(file_path):
            raise NetworkSecurityException(f"The file: {file_path} is not found",sys)     #type: ignore
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def load_numpy_array_data(file_path:str)->np.array:        #type: ignore
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            
            y_test_pred = model.predict(x_test)

            model_metric_artifact = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            report[list(models.keys())[i]] = model_metric_artifact

        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys)   #type: ignore
    