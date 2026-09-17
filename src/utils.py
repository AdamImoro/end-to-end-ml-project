import os
import pickle
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
	"""Save ``obj`` as a pickle file at ``file_path``."""
	try:
		directory = os.path.dirname(file_path)
		if directory:
			os.makedirs(directory, exist_ok=True)

		with open(file_path, "wb") as file:
			pickle.dump(obj, file)
	except Exception as e:
		raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models , params):
	try:
		report ={}
		for i in range(len(list(models.values()))):
			model = list(models.values())[i]
			model_name = list(models.keys())[i]
			param = params[model_name]

			gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
			gs.fit(X_train, y_train)

			model.set_params(**gs.best_params_)
			model.fit(X_train, y_train)

			#Make predictions
			y_train_pred = model.predict(X_train)
			y_test_pred = model.predict(X_test)

			#Evaluation 
			train_model_score = r2_score(y_train, y_train_pred)
			test_model_score = r2_score(y_test, y_test_pred)

			report[model_name] = test_model_score
		return report
	except Exception as e:
		raise CustomException(e, sys)

