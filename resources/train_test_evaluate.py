from sklearn.model_selection import \
    train_test_split, \
    GridSearchCV, \
    StratifiedKFold
from scipy.optimize import minimize_scalar
import pandas as pd
import pickle
import numpy as np

class TrainTestEvaluate:
    def __init__(self, dataframe: , pipeline, target_column,
                 param_grid=None,
                 test_size=0.25,
                 use_grid_search=True):
        self.data = dataframe
        self.pipeline = pipeline
        self.target_column = target_column
        self.param_grid = param_grid
        self.test_size = test_size
        self.use_grid_search = use_grid_search