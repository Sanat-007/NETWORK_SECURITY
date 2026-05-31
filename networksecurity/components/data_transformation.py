import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                  data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20}Data Transformation log started.{'<<'*20}")
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact

        
        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
        
    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entering get_data_transformer_object method of DataTransformation class.")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            preprocessor: Pipeline  = Pipeline(steps=[
                ("imputer", imputer)
            ])
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining training and testing file path.")
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            logging.info(f"Loading training and testing data as pandas dataframe.")
            train_df = self.read_data(valid_train_file_path)
            test_df = self.read_data(valid_test_file_path)

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_feature_train_arr = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            logging.info(f"Concatenating transformed input features and target feature for training and testing data.")
            transformed_train_arr = np.c_[transformed_input_feature_train_arr, target_feature_train_df.to_numpy()]
            transformed_test_arr = np.c_[transformed_input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=transformed_train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=transformed_test_arr)
            logging.info(f"Saving preprocessor object.")
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor_object)
            save_object(file_path="final_models/preprocessor.pkl", obj=preprocessor_object)
            logging.info(f"Data transformation completed successfully.")

            ## preparing artifact

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)   #type: ignore
        