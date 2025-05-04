import numpy as np
from src.exception import CustomException
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import logging
from dataclasses import dataclass
from src.utils import (
    RBD_method,
    PCA_method,
    load_npy,
    save_object,
    customize_time_interval,
    make_data_r_time_intervals_dependent,
)
import os
import sys
import time


@dataclass
class DataTransformationConfig:
    raw_data_arrays_path: str = os.path.join("artifacts", "raw_data_arrays")
    split_data_path: str = os.path.join("artifacts", "split_data_path.pkl")


class DataTransformation:
    def __init__(
        self,
        dim_reduction_size: int = 8,
        time_interval: float = 1,
        r: int = 1,
    ):
        self.data_transformation_config = DataTransformationConfig()
        self.dim_reduction_size = dim_reduction_size
        self.time_interval = time_interval
        self.r = r
        self.pca_time = 0
        self.rbd_time = 0

    def get_r_time_interval_dependent_data_matrices(self):
        """
        After the final data matrices are generated (for RMA, CBA, and AMA),
        some strategies require an additional step before model training.

        In this step, we enhance the dataset by reshapping data to include
        previous `r` time intervals. This allows the model to capture
        temporal correlations in the data.

        Note:
        - The `customize_time_interval` function in `utils` is used to define
        the data resolution (e.g., 30-minute, 1-hour, or 2-hour intervals).
        - The current function (this one) focuses on maping the data to a new
        space by considering `r` previous time intervals as columns.
        """

        try:
            (
                ama_data,
                cba_pca_data,
                rma_pca_data,
                rma_rbd_data,
            ) = self._get_data_matrices()

            ama_data_for_ml = make_data_r_time_intervals_dependent(ama_data, r=self.r)
            cba_data_for_ml = make_data_r_time_intervals_dependent(
                cba_pca_data, r=self.r
            )
            rma_pca_data_for_ml = make_data_r_time_intervals_dependent(
                rma_pca_data, r=self.r
            )
            rma_rbd_data_for_ml = make_data_r_time_intervals_dependent(
                rma_rbd_data, r=self.r
            )
            logging.info("Input data for ML created.")

            ama_data_for_ml = np.c_[ama_data_for_ml, ama_data[self.r:, :]]
            cba_data_for_ml = np.c_[cba_data_for_ml, ama_data[self.r :, :]]
            rma_pca_data_for_ml = np.c_[rma_pca_data_for_ml, ama_data[self.r :, :]]
            rma_rbd_data_for_ml = np.c_[rma_rbd_data_for_ml, ama_data[self.r :, :]]
            logging.info("Input data concatenated with the output for ML.")

            split_index = int(0.8 * len(ama_data_for_ml))
            train_ama, test_ama = (
                ama_data_for_ml[:split_index],
                ama_data_for_ml[split_index:],
            )
            train_cba, test_cba = (
                cba_data_for_ml[:split_index],
                cba_data_for_ml[split_index:],
            )
            train_rma_pca, test_rma_pca = (
                rma_pca_data_for_ml[:split_index],
                rma_pca_data_for_ml[split_index:],
            )
            train_rma_rbd, test_rma_rbd = (
                rma_rbd_data_for_ml[:split_index],
                rma_rbd_data_for_ml[split_index:],
            )
            logging.info("Data split into training and testing sets.")

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
            logging.error(
                f"Error in get_r_time_interval_dependent_data_matrices: {custom_error}"
            )
            raise

    def _get_data_matrices(self):
        """
        There are three modeling strategies:

            - Aggregated Model Approach (AMA):
                A single-column dataset representing the total aggregated demand
                across all users and all zip codes.

            - Cluster-Based Approach (CBA):
                A reduced-dimension matrix where each column represents the
                aggregated demand for a specific zip code across all users.

            - Reduced Model Approach (RMA):
                Applies hierarchical PCA and RBD to reduce the dimensionality of
                user demand for each zip code individually, where columns represent
                individual users.
        """

        try:
            (
                hierarchical_pca_data,
                hierarchical_rbd_data,
                data_zipcodes,
            ) = self._reduce_dimension_of_concatenated_reduced_zipcodes()
            logging.info("All zipcode data shrinked.")
            logging.info(f"PCA time: {self.pca_time:.2f} seconds")
            logging.info(f"RBD time: {self.rbd_time:.2f} seconds")

            # Data for reduced model approach (RMA)
            rma_pca_data, _ = PCA_method(
                data=hierarchical_pca_data,
                dim_reduction_size=self.dim_reduction_size,
            )
            logging.info(
                f"Formed RMA by PCA reduced from {hierarchical_pca_data.shape} to {rma_pca_data.shape}"
            )

            rma_rbd_data = RBD_method(
                data=hierarchical_rbd_data, col=self.dim_reduction_size
            )
            logging.info(
                f"Formed RMA by RBD reduced from {hierarchical_rbd_data.shape} to {rma_pca_data.shape}"
            )

            # Data for cluster-based approach (CBA)
            cba_pca_data, _ = PCA_method(
                data=data_zipcodes, dim_reduction_size=self.dim_reduction_size
            )
            logging.info(
                f"Formed CBA by PCA reduced from {data_zipcodes.shape} to {cba_pca_data.shape}"
            )

            # Data for aggregated model approach (AMA)
            ama_data = data_zipcodes.sum(axis=1).reshape(-1, 1)
            logging.info(
                f"Formed AMA reduced from {data_zipcodes.shape} to {ama_data.shape}"
            )

            return (
                ama_data,
                cba_pca_data,
                rma_pca_data,
                rma_rbd_data,
            )

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(custom_error)
            raise

    def _reduce_dimension_of_concatenated_reduced_zipcodes(self):
        hierarchical_pca_data = []
        hierarchical_rbd_data = []
        data_zipcodes = []

        try:
            for filename in os.listdir(
                self.data_transformation_config.raw_data_arrays_path
            ):
                if filename.endswith(".npy"):
                    (
                        pca_transformed_data,
                        rbd_transformed_data,
                        data_zipcode,
                    ) = self._reduce_dimension_single_zipcode(filename)

                    hierarchical_pca_data.append(pca_transformed_data)
                    logging.info(f"PCA transformed for {filename}")

                    hierarchical_rbd_data.append(rbd_transformed_data)
                    logging.info(f"RBD transformed for {filename}")

                    data_zipcodes.append(data_zipcode)
                    logging.info(f"Aggregate by zipcode transformed for {filename}")

            hierarchical_pca_data = np.hstack(hierarchical_pca_data)
            hierarchical_rbd_data = np.hstack(hierarchical_rbd_data)
            data_zipcodes = np.hstack(data_zipcodes)

            return hierarchical_pca_data, hierarchical_rbd_data, data_zipcodes

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(f"Error in getting data transformer object: {custom_error}")
            raise

    def _reduce_dimension_single_zipcode(self, filename):
        try:
            # Load the data
            data_array = load_npy(
                filename=filename,
                source=self.data_transformation_config.raw_data_arrays_path,
            )

            # customize resolution
            data_intervals = customize_time_interval(
                time_interval=self.time_interval, data=data_array
            )

            # Apply Heirarchical PCA
            start_time_pca = time.time()
            pca_transformed_data, _ = PCA_method(
                data=data_intervals, dim_reduction_size=self.dim_reduction_size
            )
            self.pca_time += time.time() - start_time_pca
            logging.info(
                f"PCA shrinked data from {data_intervals.shape} to {pca_transformed_data.shape}"
            )

            # Apply Heirarchical RBD
            start_time_rbd = time.time()
            rbd_transformed_data = RBD_method(
                data=data_intervals, col=self.dim_reduction_size
            )
            self.rbd_time += time.time() - start_time_rbd
            logging.info(
                f"RBD shrinked data from {data_intervals.shape} to {rbd_transformed_data.shape}"
            )

            # Aggregated data by zipcode
            data_zipcode = data_intervals.sum(axis=1).reshape(-1, 1)

            return pca_transformed_data, rbd_transformed_data, data_zipcode

        except Exception as e:
            custom_error = CustomException(e, sys)
            logging.error(custom_error)
            raise


# if __name__ == "__main__":
#     obj = DataTransformation()
#     obj.get_data_transformer_object()

#     data_transformation = DataTransformation()
#     data_transformation.get_data_transformer_object()
