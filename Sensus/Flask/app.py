from flask import Flask, request, render_template
from flask_cors import CORS,cross_origin
import os
import pickle 
import sklearn 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pickle.load(open('/config/workspace/feature_engineered_df.pkl','rb'))
model=pickle.load(open('/config/workspace/random_forest.pkl','rb'))
scaler=pickle.load(open('/config/workspace/scanel.pkl','rb'))
for col in df.columns:
    if df[col].dtypes == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

X = df.drop('income', axis=1)
Y = df['income']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)

app = Flask(__name__)

@app.route('/', methods=['GET'])  # route to display the Home page
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # route to show the predictions in web UI
@cross_origin()
def predict():
    if request.method == 'POST':
        # reading the inputs given by the user
        age = int(request.form['age'])
        sex = request.form['sex']
        if sex =='Female':
            sex=1
        elif sex=='Male':
            sex=0
        education_rank= int(request.form['education_rank'])
        relationship=request.form['relationship']
        if relationship=='Not-in-family':
            relationship=1
        elif relationship=='Unmarried':
            relationship=2
        elif relationship=='Husband':
            relationship=3
        elif relationship=='Wife':
            relationship=4
        elif relationship=='Own-child':
            relationship=5
        elif relationship=='Other-relative':
            relationship=6
        hours_per_week= int(request.form['hours_per_week'])
        occupation=request.form['occupation']
        if occupation=='Prof-specialty':
            occupation=1
        elif occupation=='Craft-repair':
            occupation=2
        elif occupation=='Other-service':
            occupation=3
        elif occupation=='Exec-managerial':
            occupation=4
        elif occupation=='Sales':
            occupation=5
        elif occupation=='Transport-moving':
            occupation=6
        elif occupation=='Farming-fishing':
            occupation=7
        elif occupation=='Adm-clerical':
            occupation=8
        elif occupation=='Protective-serv':
            occupation=9
        elif occupation=='Tech-support':
            occupation=10
        elif occupation=='Handlers-cleaners':
            occupation=11
        elif occupation=='Machine-op-inspct':
            occupation=12
        elif occupation=='Armed-Forces':
            occupation=13
        elif occupation=='Priv-house-serv':
            occupation=14
        capital_gain = int(request.form['capital_gain'])    
        native_country = request.form['native_country']
        if native_country=='India':
            native_country=1
        elif native_country=='United-States':
            native_country=2
        elif native_country=='Philippines':
            native_country=3
        elif native_country=='Germany':
            native_country=4
        elif native_country=='Mexico':
            native_country=5
        elif native_country=='Canada':
            native_country=6
        elif native_country=='Puerto-Rico':
            native_country=7
        elif native_country=='El-Salvador':
            native_country=8
        elif native_country=='others':
            native_country=9
       # final_features=[age,hours_per_week,capital_gain,]
        print(age,hours_per_week,capital_gain,sex,relationship,native_country,education_rank,occupation)
        features=[age,sex,education_rank,relationship,hours_per_week,occupation,capital_gain,native_country]
        int_features = [int(x) for x in features]
        final_features = [np.array(int_features)]
        prediction = model.predict(scaler.transform(final_features))
        if prediction == 1:
            output = "Income is more than 50K"
        elif prediction == 0:
            output = "Income is less than 50K"
        
    return render_template('index.html', prediction_text='{}'.format(output))

    # showing the prediction result in a UI
    return render_template('index.html', prediction_text='{}'.format(output))



if __name__ == '__main__':
    app.run(debug=True,port=5001,host='0.0.0.0')