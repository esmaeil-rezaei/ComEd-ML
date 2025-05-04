import os
import sys
import numpy as np
import pandas as pd
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


class ModelTrainer:
    def __init__(self, strategy_name: str, ml_models_name: set):
        self.trained_model_file_path = os.path.join(
            "artifacts", f"model_{strategy_name}.pkl"
        )
        self.strategy_name = strategy_name
        self.ml_models_name = ml_models_name

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
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
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
            logging.error(f"Error in train method: {custom_error}")
            raise custom_error
