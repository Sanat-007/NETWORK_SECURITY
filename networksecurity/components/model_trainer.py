import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

import mlflow
from dotenv import load_dotenv
load_dotenv()
repo_owner=os.getenv("repo_owner")
repo_name=os.getenv("repo_name")
import dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
    
    def track_mlflow(self,best_model,train_classification_metrics):
        try:
            with mlflow.start_run():
                f1_score = train_classification_metrics.f1_score
                precision_score = train_classification_metrics.precision_score
                recall_score = train_classification_metrics.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.sklearn.log_model(       #type: ignore
                    sk_model=best_model,
                    name="model"
                )
                

        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
        
    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(verbose=1),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
            }

            params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'],
                
                'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
             
            }

            model_report:dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)

            ## To get best model score from dict
            best_model_name = max(
                model_report,
                key=lambda model_name: model_report[model_name].f1_score
            )

            best_model = models[best_model_name]

            best_model_score = model_report[best_model_name].f1_score
            
            y_train_pred = best_model.predict(x_train)

            train_classification_metrics = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            
            ## Track the experiment in mlflow
            self.track_mlflow(best_model,train_classification_metrics)

            y_test_pred = best_model.predict(x_test)
            test_classification_metrics = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            self.track_mlflow(best_model,test_classification_metrics)

            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessor_object_file_path)

            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir,exist_ok=True)

            Network_model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_model)

            save_object(file_path="final_models/model.pkl", obj=Network_model)

            ## Model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,train_metric_artifact=train_classification_metrics,test_metric_artifact=test_classification_metrics)  #type: ignore
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training data")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            test_array = load_numpy_array_data(test_file_path)
            train_array = load_numpy_array_data(train_file_path)

            logging.info("Splitting training data into X and y")
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Training model")
            model_trainer_artifact = self.train_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore

