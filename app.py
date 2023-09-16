from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
import pandas as pd
import model as ml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

encoded_cols = ml.encoded_cols
categorical_cols = ml.categorical_cols
cols_scaled = ml.cols_scaled
encoder = ml.encoder
minscaler = ml.minscaler


app = Flask(__name__)

model= pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features=[x for x in request.form.values()]
        finalarray=[np.array(int_features)]
        def isFloat(s):
            try:
                float(s)
                return True
            except:
                return False
            

        st = str(finalarray[0][0])
        loc = str(finalarray[0][4])
        k = 42
        if (st == 'male') or (st == 'female'):
            k = 1
        if k == 42:
            return render_template('index.html',pred='Invalid gender description!! please write male/female')
        
        if not isFloat(finalarray[0][1]):
            return render_template('index.html',pred='Invalid age please write integer value')
        
        if not isFloat(finalarray[0][2]):
            return render_template('index.html',pred='Invalid input')
        
        if not isFloat(finalarray[0][3]):
            return render_template('index.html',pred='Invalid input')
        
        if loc not in ["Chicago", "Miami", "Houston", "Los Angeles", "New York"]:
            return render_template('index.html',pred='Invalid location')
        
        if not isFloat(finalarray[0][5]):
            return render_template('index.html',pred='Invalid input!')
    
        def input(finalarray):
            final = {
            "Gender" : finalarray[0][0],
            "Age"  : finalarray[0][1],
            "Subscription_Length_Months" : finalarray[0][2],
            "Monthly_Bill" : finalarray[0][3],
            "Location" : finalarray[0][4],
            "Total_Usage_GB" : finalarray[0][5],
            "Total_Amount_Charged" : float(finalarray[0][2]) * float(finalarray[0][3])

            } 
            new_input_df = pd.DataFrame([final])
            new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
            new_input_df[cols_scaled] = minscaler.transform(new_input_df[cols_scaled])
            x_input = new_input_df[cols_scaled + encoded_cols]
            return x_input 
        x_input = input(finalarray)
        prediction=model.predict(x_input)
        output= prediction[0]

        if output == 1:
            return render_template('index.html',pred='Prediction is 1')
        else:
            return render_template('index.html',pred='Prediction is 0')


    else:
        return render_template("index.html")


if __name__ == "__main__" :
    app.run(debug=True)