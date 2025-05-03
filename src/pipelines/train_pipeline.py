import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_pipeline import DataPipeline
from src.components.model_trainer import ModelTrainer
from src.utils import get_data_attribute


class TrainStrategiesModel():
    """
    In this module, we find train models for all strategies (AMA, CBA, RMA).

    - dim_reduction_size: For the RMA and CBA we use dimensionality redction.
    This paramter determines the number of columns after reduction.
    - time_interval: It determines the resolution by hour. This step is done
    before dimensionality reduction. This is the very initial step. Options are
    0.5, 1, 2
    - r: It stands for the number of previous time intervals to correlate the
    current value to the r previous values.
    - strategy_name: Used to name the model trainer for each strategy.
    """

    def __init__(
        self,
        dim_reduction_size: int,
        time_interval: float,
        strategy_name: dict,
        ml_models_name: dict,
        r: int,
    ):
        self.dim_redcution_size = dim_reduction_size
        self.time_interval = time_interval
        self.strategy_name = strategy_name
        self.ml_models_name = ml_models_name
        self.r = r

    def train(self):
        try:
            data_transformation = DataTransformation(
                dim_reduction_size=self.dim_redcution_size,
                time_interval=self.time_interval,
                r=self.r,
            )

            (
                train_ama,
                test_ama,
                train_cba,
                test_cba,
                train_rma_pca,
                test_rma_pca,
                train_rma_rbd,
                test_rma_rbd,
            ) = data_transformation.get_r_time_interval_dependent_data_matrices()

            data_pipeline = DataPipeline()
            data_pipeline.initiate_data_all_pipelines(
                train_ama,
                test_ama,
                train_cba,
                test_cba,
                train_rma_pca,
                test_rma_pca,
                train_rma_rbd,
                test_rma_rbd,
            )

            data_names = {
                "train_ama": train_ama,
                "test_ama": test_ama,
                "train_cba": train_cba,
                "test_cba": test_cba,
                "train_rma_pca": train_rma_pca,
                "test_rma_pca": test_rma_pca,
                "train_rma_rbd": train_rma_rbd,
                "test_rma_rbd": test_rma_rbd,
            }

            model_trainer = ModelTrainer(
                strategy_name = self.strategy_name,
                ml_models_name = self.ml_models_name
            )

            r2_square = model_trainer.train(
                data_train=data_names[f"train_{self.strategy_name}"],
                data_test=data_names[f"test_{self.strategy_name}"],
            )
            print(r2_square)
        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in TrainStrategiesModel: {custom_error}")


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

    TrainStrategiesModel = TrainStrategiesModel(
        dim_reduction_size=8,
        time_interval=1,
        strategy_name="rma_rbd",
        ml_models_name={"Linear Regression"},
        r=3,
    )
    TrainStrategiesModel.train()
