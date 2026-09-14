import os
import pickle
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np


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
