import pandas as pd 
import numpy as np
from Sensus.exception import SensusException
from Sensus.logger import logging
import data_dump
import pickle
import os,sys
from Sensus.DataCleaning.data_cleaning import DataCleaning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from feature_engine.imputation import CategoricalImputer
class ModelBuilding:
    def __init__(self):
        pass
    def start_model_building(self):
        try:
            #loading dataframe after feature Engineering 
            with open('/config/workspace/feature_engineered_df.pkl','rb') as f:
                df=pickle.load(f)
        #lable encoding to our target feature income 
        #lets do train test spilt
            target = 'income'
            features = df.columns.tolist()
            features.remove(target)
            X = df[features]
            y = df[target]
        #labe encoding 
            y=LabelEncoder().fit_transform(y)
            num_features=[ col for col in X.columns if X[col].dtypes!='O']
            cat_features=[ col for col in X.columns if X[col].dtypes=='O']
            logging.info(f"Numerical features in dataframe :{num_features}")
            logging.info(f"Categorical features in dataframe:{cat_features}")
           
        #Using robust scaler to scale numerical features and to treat outliers 
        #One hot encoding to categorical features 
            for col in X.columns:
                if X[col].dtypes=='object':
                    encoder=LabelEncoder()
                    X[col]=encoder.fit_transform(X[col])
        #scaling data 
            logging.info("scalling features")
            scaler=StandardScaler()
            X=scaler.fit_transform(X)
            #handling imbalanced dataset 
            logging.info("Handling imbalanced dataset")
            ran_over=RandomOverSampler(random_state=42)
            X_resampled,y_resampled=ran_over.fit_resample(X,y)
            #X=preprocessor2.fit_transform(X)
            X_train_resampled,X_test_resampled,y_train_resampled,y_test_resampled=train_test_split(X_resampled,y_resampled,random_state=42,test_size=0.2)
            logging.info("Splitting data into train test split")
            logging.info(f"Shape of Train dataset:{X_train_resampled.shape}")
            logging.info(f"Shape of Test dataset:{X_test_resampled.shape}")
            #trying to build model
            logging.info("Model building started")
            random_forest=RandomForestClassifier(n_estimators=74,max_depth=95,random_state=30)
            random_forest.fit(X_train_resampled,y_train_resampled)
            logging.info(f"accuracy score of Proposed model is :{accuracy_score(y_test_resampled,random_forest.predict(X_test_resampled))}")
            print(df.columns)
            #print(random_forest.predict(scaler.transform(np.array([[31,16,2,1,0,14,2,2]]))))
            #Applying GrfidSearchCV to tune Hyperparameter 
            logging.info("Dumping scaler and encoder  object ")
            logging.info("Dumping dataframe and model object ")
            pickle.dump(df, open("dataframe.pkl","wb"))
            logging.info("Dumping scaler object")
            pickle.dump(scaler,open("scanel.pkl","wb"))
            pickle.dump(random_forest,open('random_forest.pkl','wb'))
        except Exception as e:
            raise SensusException(e,sys)
m=ModelBuilding()
m.start_model_building()