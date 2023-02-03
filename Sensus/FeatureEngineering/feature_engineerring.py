import pandas as pd 
import numpy as np
from Sensus.exception import SensusException
from Sensus.logger import logging
import data_dump
import pickle
import os,sys
from Sensus.DataCleaning.data_cleaning import DataCleaning
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class FeatureEngineering:
    def __init__(self):
        pass 
    def feature_engineering(self):
        try:
             #loading cleaned dataset from cleandataset
            with open("/config/workspace/clean_df.pkl",'rb') as f:
                df=pickle.load(f)
            logging.info(f"\n \n Null values inside the dataset->:{df.isnull().sum()}")
            #Since our dataset contain null values in the form of '?' first we have to fill it py np.nan
            df.replace({'?':np.NaN},inplace=True)
            logging.info(f"Null values inside dataset-->{df.isnull().sum().sort_values(ascending=False)}")
            #finding marital_status_unique values()
            logging.info("Replacing overall marital status into Single or Married")
            #Replacing  overall status of marital.status to single or married
            df.replace(to_replace=df['marital_status'].unique(),
                    value = ['single','married','single','single','single','married','single'], inplace=True)
            logging.info("new dataframe")
            logging.info(df.head(2))
            df_education_labels = df.groupby(by = 'education').describe()['education_num']['mean'].sort_values().reset_index()
            df.rename(columns = {'education_num': 'education_rank'}, inplace=True)
            # naming less frequent countries as others (having value counts less than 0.3% of total values)
            percentage_threshold = 0.3
            arr_others = df['native_country'].value_counts()[df['native_country'].value_counts(normalize=True)*100 < percentage_threshold].index
            df['native_country'].replace(to_replace=arr_others, value = ['others']*len(arr_others), inplace=True)
            logging.info("*********************************************************************************")
            logging.info(df.head(2))
            logging.info("************************************************************************************")
            logging.info("****************************************************************************")
            logging.info("Filling missing values of categorical features by mode ")
            columns_with_nan = ['workclass', 'occupation', 'native_country']
            df['workclass']=df['workclass'].fillna(df['workclass'].mode()[0])
            df['occupation']=df['occupation'].fillna(df['occupation'].mode()[0])
            df['native_country']=df['native_country'].fillna(df['native_country'].mode()[0])
            logging.info("Dropping features which are not relevant for our analysis(fnlwgt,education,race,marital_status,race,native_country,capital_loass")
            df=df.drop(columns=['fnlwgt','workclass','marital_status','education','race','capital_loss'],axis=1)
            pickle.dump(df,open("feature_engineered_df.pkl","wb"))
        except Exception as e:
            raise SensusException(e,sys)
f=FeatureEngineering()
f.feature_engineering()