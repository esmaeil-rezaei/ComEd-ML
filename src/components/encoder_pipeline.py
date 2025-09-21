import os
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class EncoderPipelineConfig:
    preprocessor_ama_obj_path = os.path.join(f"models", "preprocessor_ama.pkl")
    preprocessor_cba_obj_path = os.path.join(f"models", "preprocessor_cba.pkl")
    preprocessor_rma_pca_obj_path = os.path.join(f"models", "preprocessor_rma_pca.pkl")
    preprocessor_rma_rbd_obj_path = os.path.join(f"models", "preprocessor_rma_rbd.pkl")


class EncoderPipeline:
    def __init__(
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
        
        self.train_ama = train_ama
        self.test_ama = test_ama
        self.train_cba = train_cba
        self.test_cba = test_cba
        self.train_rma_pca = train_rma_pca
        self.test_rma_pca = test_rma_pca
        self.train_rma_rbd = train_rma_rbd
        self.test_rma_rbd = test_rma_rbd
        self.encoder_pipeline_config = EncoderPipelineConfig()


    def encode_all_data_for_all_strategies(
        self,
    ):

        try:
            (
                train_ama,
                test_ama,
                preprocessing_obj_ama,
            ) = self._encode_standardize_single_strategy(
                train_data=self.train_ama,
                test_data=self.test_ama,
            )
            logging.info("Preprocessing pipeline done for ama")

            (
                train_cba,
                test_cba,
                preprocessing_obj_cba,
            ) = self._encode_standardize_single_strategy(
                train_data=self.train_cba,
                test_data=self.test_cba,
            )
            logging.info("Preprocessing pipeline done for cba")

            (
                train_rma_pca,
                test_rma_pca,
                preprocessing_obj_rma_pca,
            ) = self._encode_standardize_single_strategy(
                train_data=self.train_rma_pca,
                test_data=self.test_rma_pca,
            )
            logging.info("Preprocessing pipeline done for rma_pca")

            (
                train_rma_rbd,
                test_rma_rbd,
                preprocessing_obj_rma_rbd,
            ) = self._encode_standardize_single_strategy(
                train_data=self.train_rma_rbd,
                test_data=self.test_rma_rbd,
            )

            logging.warning("Preprocessing pipeline done for rma_rbd")

            save_object(
                file_path=self.encoder_pipeline_config.preprocessor_ama_obj_path,
                obj={
                    "train_ama": train_ama,
                    "test_ama": test_ama,
                    "preprocessing_obj_ama": preprocessing_obj_ama,
                }
            )

            save_object(
                file_path=self.encoder_pipeline_config.preprocessor_cba_obj_path,
                obj={
                    "train_cba": train_cba,
                    "test_cba": test_cba,
                    "preprocessing_obj_cba": preprocessing_obj_cba,
                }
            )

            save_object(
                file_path=self.encoder_pipeline_config.preprocessor_rma_pca_obj_path,
                obj={
                    "train_rma_pca": train_rma_pca,
                    "test_rma_pca": test_rma_pca,
                    "preprocessing_obj_rma_pca": preprocessing_obj_rma_pca,
                }
            )

            save_object(
                file_path=self.encoder_pipeline_config.preprocessor_rma_rbd_obj_path,
                obj={
                    "train_rma_rbd": train_rma_rbd,
                    "test_rma_rbd": test_rma_rbd,
                    "preprocessing_obj_rma_rbd": preprocessing_obj_rma_rbd,
                }
            )

            logging.info(f"Saved preprocessing objects.")

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
            logging.error({custom_error})
            raise


    def _encode_standardize_single_strategy(self, train_data, test_data):
        try:
            preprocessing_obj = self._get_encoding_pipeline_object(train_data)

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
            logging.error({custom_error})
            raise


    def _get_encoding_pipeline_object(self, data: np.ndarray):
        """
        Normally, we use:

            numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = data.select_dtypes(include=["object", "category"]).columns

        to automatically detect numerical and categorical features 
        (Pandas DataFrame). Pandas is usually preferred in this step 
        because **column names matter** when selecting features.

        However, in this study, the columns in our data are the 
        **reduced features** (either eigenvectors or basis vectors), 
        so they **do not have meaningful names**. We also do not 
        have categorical features.

        Therefore, we explicitly set:

            categorical_columns = []

        and drop the last numerical column from numerical_columns 
        (since it represents the target variable).

        ⚠️ Be careful: this assumes your data matches this structure. 
        If your dataset contains categorical features or a different 
        column arrangement, you will need to modify this part accordingly.
        """


        try:
            # numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
            # categorical_columns = data.select_dtypes(include=["object", "category"]).columns

            numerical_columns = range(data.shape[1] - 1)
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
            logging.error(custom_error)
            raise

