import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import os 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass 
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()


    def get_data_transformation_object(self):
        """This function is for data transformation."""

        try:
            numeric_columns = ["writing_score", "reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed.")
            logging.info("Categorical columns encoding completed.")
            preprocessor = ColumnTransformer([
                ("num_pipeeline", num_pipeline, numeric_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def intiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Read of train and test completed!")
            logging.info("Obtaining preprocessor objects")

            preprocessing_object = self.get_data_transformation_object()
            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing object.")

            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj =preprocessing_object
            )

            return [
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ]
        except Exception as e:
            raise CustomException(e, sys)

