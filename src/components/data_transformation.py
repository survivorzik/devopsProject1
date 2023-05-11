# import sys 
# from dataclasses import dataclass
# from src.logger import logging
# from src.exception import CustomException
# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split    
# import os
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()
#         self.logger=logging.getLogger(__name__)
    
#     def get_data_transformer_object(self):
#         try:
#             self.logger.info("Initiating data transformation")
#             # self.logger.info("Reading data")
#             # df=pd.read_csv("./notebook/data/stud.csv")
#             self.logger.info("Creating preprocessor object")
#             num_features = ["writing_score","reading_score"]
#             cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
#             num_transformer = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())])
#             cat_transformer = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder()),
#                 ('scaler',StandardScaler())])
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('num', num_transformer, num_features),
#                     ('cat', cat_transformer, cat_features)])
            
#             # self.logger.info("Fitting preprocessor object")
#             # preprocessor.fit(df)
#             # self.logger.info("Saving preprocessor object")
#             # os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),exist_ok=True)
#             # pd.to_pickle(preprocessor,self.data_transformation_config.preprocessor_obj_file_path)
#             # self.logger.info("Data transformation completed")
#             return preprocessor
#         except Exception as e:
#             self.logger.error("Error in data transformation")
#             self.logger.error(e)
#             raise CustomException(e,sys)    
#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)
#             self.logger.info("Read Train and Test Data Completed")
#             self.logger.info("Obtaining preprocessing object ")
#             preprocessor_obj=self.get_data_transformer_object()
#             target_column_name="math_score"
#             num_columns=["writing_score","reading_score"]
            
#             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
#             target_feature_train_df=train_df[target_column_name]
            
#             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
#             target_feature_test_df=test_df[target_column_name]
            
#             self.logger.info(f"Apply preprocessing object on training dataframe and testing dataframe.")
            
#             input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
#             train_arr=np.c_[
#                 input_feature_train_arr,np.array(target_feature_train_df)
#             ]
#             test_arr=np.c_[
#                 input_feature_test_arr,np.array(target_feature_test_df)
#             ]
#             self.logger(f'Preprocessor Saved')
#             save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)
#             return(
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )
            
#         except Exception as e:
#             self.logger.error("Error in data transformation")
#             self.logger.error(e)
#             raise CustomException(e,sys)


# # if __name__=="__main__":
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
