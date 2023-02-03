import pymongo
import pandas as pd
import json

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATABASE_NAME="Sensus"
COLLECTION_NAME="income"
DATA_FILE_PATH="/config/workspace/adult.csv"

if __name__=="__main__":
    df=pd.read_csv(DATA_FILE_PATH)
    print(f" Rows and column in dataframe ,{df.shape}")

    #Convert dataframe into json so that we can dump into mongodb
    df.reset_index(drop=True,inplace=True)
    json_record=list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)