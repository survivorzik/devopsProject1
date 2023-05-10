import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.logger = logging.getLogger(__name__)
    def initiate_data_ingestion(self):
        try:
            self.logger.info("Initiating data ingestion")
            self.logger.info("Reading raw data")
            # raw_data = pd.read_csv(self.ingestion_config.raw_data_path)
            df=pd.read_csv("./notebook/data/stud.csv")
            self.logger.info("Splitting data into train and test")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            self.logger.info("Saving raw data")
            df.to_csv(self.ingestion_config.raw_data_path, index=False , header=True)
            self.logger.info("Saving train and test data")
            train_data.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            self.logger.info("Data ingestion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            self.logger.error("Error in data ingestion")
            self.logger.error(e)
            raise CustomException(e,sys)
        
if __name__=="__main__":
    di = DataIngestion()
    di.initiate_data_ingestion()