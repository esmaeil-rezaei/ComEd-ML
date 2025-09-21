import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = Path("models")


class ModelTrainer:
    def __init__(
            self,
            strategy_name: str,
            ml_models_name: set,
            time_interval: float,
            r: int,):

        self.strategy_name = strategy_name
        self.ml_models_name = ml_models_name
        self.model_trainer_config = ModelTrainerConfig()

        if time_interval not in {0.5, 1.0, 2.0}:
            logging.error(
                f"Invalid time_interval: {time_interval}. Must be one of 0.5, 1.0, or 2.0"
            )

        self.trained_model_file_path = os.path.join(
            self.model_trainer_config.trained_model_file_path,
            f"model_{strategy_name}.pkl"
        )

    def train(self, data_train: np.ndarray, data_test: np.ndarray):
        try:
            logging.info("Split training and test input data")

            X_train, X_test, y_train, y_test = (
                data_train[:, :-1],
                data_test[:, :-1],
                data_train[:, -1],
                data_test[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'max_features': ['auto', 'sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'max_features': ['sqrt', 'log2', None],
                    # 'bootstrap': [True, False]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'max_depth': [3, 5, 7],
                    # 'min_samples_split': [2, 5],
                    # 'min_samples_leaf': [1, 2],
                    # 'subsample': [0.7, 0.9, 1.0],
                    # 'max_features': ['sqrt', 'log2']
                },
                "Linear Regression": {
                    # 'fit_intercept': [True, False],
                    # 'positive': [True, False]
                },
                "XGBRegressor": {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    # 'max_depth': [3, 5, 7],
                    # 'min_child_weight': [1, 3, 5],
                    # 'gamma': [0, 1, 5],
                    # 'subsample': [0.7, 0.9, 1.0],
                    # 'colsample_bytree': [0.7, 0.9, 1.0],
                    # 'reg_alpha': [0, 0.01, 0.1],
                    # 'reg_lambda': [1, 1.5, 2.0],
                },
                "CatBoosting Regressor": {
                    'iterations': [300, 500, 800],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    # 'l2_leaf_reg': [1, 3, 5, 7],
                    # 'bagging_temperature': [0, 1, 3],
                    # 'border_count': [32, 64, 128]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 300],
                    'learning_rate': [0.01, 0.1, 1],
                    # 'loss': ['linear', 'square', 'exponential']
                },
            }

            models_selected = {name: models[name] for name in self.ml_models_name}
            params_selected = {name: params[name] for name in self.ml_models_name}

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models_selected,
                param=params_selected,
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.trained_model_file_path,
                obj={
                    "best_model_name": best_model_name,
                    "best_model": best_model,
                    f"X_train_{self.strategy_name}": X_train,
                    f"X_test_{self.strategy_name}": X_test,
                    f"y_train_{self.strategy_name}": y_train,
                    f"y_test_{self.strategy_name}": y_test,
                }
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square, best_model_name

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error({custom_error})
            raise
