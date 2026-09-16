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
from xgboost import XGBRFRegressor
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
                "Xgboost": XGBRFRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

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
