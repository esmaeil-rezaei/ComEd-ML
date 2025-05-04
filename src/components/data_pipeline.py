import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataPipelineConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataPipeline:
    def __init__(self):
        self.data_pipeline_config = DataPipelineConfig()

    def initiate_data_all_pipelines(
        self,
        train_ama: np.ndarray,
        test_ama: np.ndarray,
        train_cba: np.ndarray,
        test_cba: np.ndarray,
        train_rma_pca: np.ndarray,
        test_rma_pca: np.ndarray,
        train_rma_rbd: np.ndarray,
        test_rma_rbd: np.ndarray,
    ):
        try:
            (
                train_ama,
                test_ama,
                preprocessing_obj_ama,
            ) = self._initiate_data_single_pipeline(
                train_data=train_ama,
                test_data=test_ama,
            )
            logging.info("Preprocessing pipeline done for ama")

            (
                train_cba,
                test_cba,
                preprocessing_obj_cba,
            ) = self._initiate_data_single_pipeline(
                train_data=train_cba,
                test_data=test_cba,
            )
            logging.info("Preprocessing pipeline done for cba")

            (
                train_rma_pca,
                test_rma_pca,
                preprocessing_obj_rma_pca,
            ) = self._initiate_data_single_pipeline(
                train_data=train_rma_pca,
                test_data=test_rma_pca,
            )
            logging.info("Preprocessing pipeline done for rma_pca")

            (
                train_rma_rbd,
                test_rma_rbd,
                preprocessing_obj_rma_rbd,
            ) = self._initiate_data_single_pipeline(
                train_data=train_rma_rbd,
                test_data=test_rma_rbd,
            )
            logging.info("Preprocessing pipeline done for rma_rbd")

            save_object(
                file_path=self.data_pipeline_config.preprocessor_obj_path,
                obj={
                    "train_ama": train_ama,
                    "test_ama": test_ama,
                    "preprocessing_obj_ama": preprocessing_obj_ama,
                    "train_cba": train_cba,
                    "test_cba": test_cba,
                    "preprocessing_obj_cba": preprocessing_obj_cba,
                    "train_rma_pca": train_rma_pca,
                    "test_rma_pca": test_rma_pca,
                    "preprocessing_obj_rma_pca": preprocessing_obj_rma_pca,
                    "train_rma_rbd": train_rma_rbd,
                    "test_rma_rbd": test_rma_rbd,
                    "preprocessing_obj_rma_rbd": preprocessing_obj_rma_rbd,
                },
            )
            logging.info(f"Saved preprocessing object.")
            return (
                train_ama,
                test_ama,
                train_cba,
                test_cba,
                train_rma_pca,
                test_rma_pca,
                train_rma_rbd,
                test_rma_rbd,
            )

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in initiate_data_all_pipelines: {custom_error}")
            raise custom_error

    def _initiate_data_single_pipeline(self, train_data, test_data):
        try:
            preprocessing_obj = self._get_data_pipeline_object(train_data)

            input_train_data = train_data[:, :-1]
            target_train_data = train_data[:, -1]

            input_test_data = test_data[:, :-1]
            target_test_data = test_data[:, -1]

            input_train = preprocessing_obj.fit_transform(input_train_data)
            input_test = preprocessing_obj.transform(input_test_data)

            train_arr = np.c_[input_train, np.array(target_train_data)]
            test_arr = np.c_[input_test, np.array(target_test_data)]

            return (
                train_arr,
                test_arr,
                preprocessing_obj,
            )

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in initiate_data_pipeline: {custom_error}")

    def _get_data_pipeline_object(self, data: np.ndarray):
        """
        This function is responsible for data trnasformation

        """
        try:
            numerical_columns = np.arange(data.shape[1] - 1)
            categorical_columns = []

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in DataPipeline: {custom_error}")
