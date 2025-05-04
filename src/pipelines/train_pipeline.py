import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_pipeline import DataPipeline
from src.components.model_trainer import ModelTrainer
import cProfile

class TrainStrategiesModel:
    """
    In this module, we train models for different strategies (AMA, CBA, RMA).

    Parameters:
    - dim_reduction_size: Defines the number of columns after dimensionality reduction for RMA and CBA.
    - time_interval: Resolution by hour before dimensionality reduction. Options are 0.5, 1, 2.
    - r: Number of previous time intervals to correlate with the current value.
    - strategy_names: Name of the strategy used for model training.
    - ml_models_name: A dictionary containing the model names for each strategy.
    """

    def __init__(
        self,
        dim_reduction_size: int,
        time_interval: float,
        strategy_names: set,
        ml_models_name: set,
        r: int,
    ):
        self.dim_reduction_size = dim_reduction_size
        self.time_interval = time_interval
        self.strategy_names = strategy_names
        self.ml_models_name = ml_models_name
        self.r = r

    def train(self,):
        try:
            # Prepare data for training
            data_transformation = DataTransformation(
                dim_reduction_size=self.dim_reduction_size,
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

            # Standardize Numericals & Encode Categoricals
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

            # Train models for each strategy
            for strategy in self.strategy_names:
                model_trainer = ModelTrainer(
                    strategy_name=strategy, ml_models_name=self.ml_models_name
                )

                r2_square, best_model_name = model_trainer.train(
                    data_train=data_names[f"train_{strategy}"],
                    data_test=data_names[f"test_{strategy}"],
                )
                logging.info(
                    f"R2 score for model {best_model_name} and trategy {strategy}: {r2_square}"
                )

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in TrainStrategiesModel: {custom_error}")


if __name__ == "__main__":
    
    profiler = cProfile.Profile()
    profiler.enable()

    wants_data_ingestion = False
    try:
        if wants_data_ingestion == True:
            data_ingestion = DataIngestion()
            data_ingestion.initiate_data_ingestion()
        else:
            logging.info("Data ingestion skipped.")

        trainer = TrainStrategiesModel(
            dim_reduction_size=8,
            time_interval=1,
            strategy_names={"rma_rbd", "rma_pca"},
            ml_models_name={"Linear Regression"},
            r=3,
        )
        trainer.train()

        profiler.disable()
        profiler.print_stats(sort='time')

    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in main: {custom_error}")
