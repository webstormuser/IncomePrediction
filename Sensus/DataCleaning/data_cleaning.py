import pandas as pd 
import numpy as np 
from Sensus.exception import SensusException
from Sensus.logger import logging
from Sensus.config import mongo_client
from utils import get_collection_as_dataframe
import pickle
import data_dump
import os,sys
import warnings
warnings.filterwarnings("ignore")
database_name='Sensus'
collection_name='income'
clean_df='pickle.dat'

class DataCleaning:
    def __init__(self):
        pass
    def start_cleaning(self):
        try:
            '''
                Project :Sensus Income Prediction
                Authour:Ashwini
                This function collects data from collection from MongoDB database and stored it in the form of Pandas Dataframe.
                From the Dataframe it removes extra spaces fro features and remove duplicate records from the dataset 
                and replace . from column names by '_'
            '''
            df=get_collection_as_dataframe(database_name, collection_name)
            logging.info("Deleting Extra spaces from the columns in the dataset")
            df=df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            logging.info(f'duplicate records inside dataset->,{df.duplicated().sum()}')
            logging.info(f'deleting duplicate records')
            df.drop_duplicates(inplace=True)
            logging.info(f'Dupicate records inside dataset :{df.duplicated().sum()}')
            logging.info(f'shape of dataset:{df.shape}')
            logging.info("Replacing space in coulmn name by _ ")
            df = df.rename(columns=lambda x: x.replace('.', '_'))
            logging.info(f"saving clean dataset")
            pickle.dump(df,open("clean_df.pkl","wb"))
        except Exception as e :
            raise SensusException(e,sys)
d=DataCleaning()
d.start_cleaning()
   