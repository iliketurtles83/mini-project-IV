# Flask app to predict loan status using our pickled model from LoanPredictions.py

# import Flask and jsonify
from flask import Flask, request, jsonify

# import numpy
import numpy as np

# import pandas
import pandas as pd

# import Resource, Api and reqparser
from flask_restful import Api, Resource, reqparse

# import pickle
import pickle

app = Flask(__name__)
api = Api(app)

# load our pickled model
model = pickle.load(open('model.pkl', 'rb'))

# json post route
@app.route('/predictLoan', methods=['POST'])
def predictLoan():
    # get data from post request
    req_data = request.get_json(force=True)
    if req_data:
        gender = req_data['Gender']
        married = req_data['Married']
        dependents = req_data['Dependents']
        education = req_data['Education']
        self_employed = req_data['Self_Employed']
        applicant_income = req_data['ApplicantIncome']
        coapplicant_income = req_data['CoapplicantIncome']
        loan_amount = req_data['LoanAmount']
        loan_amount_term = req_data['Loan_Amount_Term']
        credit_history = req_data['Credit_History']
        property_area = req_data['Property_Area']

    #create TotalIncome column as a sum of ApplicantIncome and CoapplicantIncome
    TotalIncome = applicant_income + coapplicant_income
    
    #create TotalIncome_log column as a log of TotalIncome
    TotalIncome_log = np.log(TotalIncome)
    #create LoanAmount_log
    LoanAmount_log = np.log(loan_amount)

    # create a dict of features to be fed into our model
    features = {
        'Gender' : [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'TotalIncome_log': [TotalIncome_log],
        'LoanAmount_log': [LoanAmount_log]
    }

    # create a dataframe from the dict
    features_df = pd.DataFrame(features)

    # predict using our model
    y_pred = model.predict(features_df)

    return jsonify({'prediction' : y_pred[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)