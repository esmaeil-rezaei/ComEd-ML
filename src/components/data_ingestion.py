import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import unzip_file, clean_reshape_files_to_npy
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration
    """

    raw_data_zipped_files_path: str = os.path.join("artifacts", "raw_data_zipped")
    raw_data_unzipped_files_path: str = os.path.join("artifacts", "raw_data_unzipped")
    raw_data_arrays_path: str = os.path.join("artifacts", "raw_data_arrays")


class DataIngestion:
    """
    Data Ingestion Class
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiate Data Ingestion
        """

        logging.info("Data Ingestion started")

        try:
            # Unzipe data
            unzip_file(
                source=self.data_ingestion_config.raw_data_zipped_files_path,
                dest=self.data_ingestion_config.raw_data_unzipped_files_path,
            )
            logging.info("Unzipping completed")

            # Data wrangling
            clean_reshape_files_to_npy(
                source=self.data_ingestion_config.raw_data_unzipped_files_path,
                dest=self.data_ingestion_config.raw_data_arrays_path,
            )
            logging.info("Data wrangling completed")

        except Exception as e:
            raise CustomException(e, sys) from e

