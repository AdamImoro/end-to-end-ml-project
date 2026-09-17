import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Start splitting test and train array into train and test set!')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models ={
                "Adaboost Regressor": AdaBoostRegressor(),
                "Gradientbost Regressor": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Kneigbours": KNeighborsRegressor(),
                "Catboost": CatBoostRegressor(),
                "Xgboost": XGBRegressor()
            }
            params = {
                "Adaboost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                },
                "Gradientbost Regressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.7, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Kneigbours": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]  # 1 = manhattan, 2 = euclidean
                },
                "Catboost": {
                    "iterations": [100, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },
                "Xgboost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # The best model score 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model')
            logging.info(f"Best found model on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.model_trainer_obj_file_path,
                obj=best_model
            )

            predictions = best_model.predict(X_test)
            r_score = r2_score(y_test, predictions)

            return r_score
        except Exception as e:
            raise CustomException(e, sys)
