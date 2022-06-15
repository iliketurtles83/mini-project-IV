#import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

#import data
df=pd.read_csv('data.csv')

#create TotalIncome column as a sum of ApplicantIncome and CoapplicantIncome
df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']

#create TotalIncome_log column as a log of TotalIncome
df['TotalIncome_log']=np.log(df['TotalIncome'])

#create LoanAmount_log column
df['LoanAmount_log']=np.log(df['LoanAmount'])

df.drop(columns=['ApplicantIncome','CoapplicantIncome','LoanAmount', 'TotalIncome'],inplace=True)
'''
# create class LogDfTransform that takes care of outliers using log transformation
class LogDfTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columnNames):
        self.columnNames = columnNames
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X=X.copy()
        X.loc[:,self.columnNames]=np.log(X[self.columnNames]).values
        return X

# income_log does the log transformation of TotalIncome and LoanAmount
income_log = LogDfTransform(['ApplicantIncome','LoanAmount'])
'''
# create pipelines for numerical and categorical columns
# pipeline for numerical columns (log transform -> imputation -> standard scaler -> selectkbest)
numerical_transform = Pipeline([('impute_mean', SimpleImputer(strategy='mean')),
                                ('scaling', StandardScaler()),
                                ('select_kbest', SelectKBest(k=3))])

# pipeline for categorical columns
categorical_transform = Pipeline([('impute_mode', SimpleImputer(strategy='most_frequent')),
                                ('one-hot-encode', OneHotEncoder(handle_unknown = 'ignore'))])

# columntransformer for numerical and categorical columns
preprocessing_df = ColumnTransformer([('numerical', numerical_transform, ['TotalIncome_log', 'LoanAmount_log','Loan_Amount_Term','Credit_History']),
('categorical', categorical_transform, ['Gender', 'Married', 'Dependents', 'Education', 
'Self_Employed', 'Property_Area'])])

# create a LogisticRegression classifier
logistic = LogisticRegression(max_iter=10000)

# build a pipeline for our model
pipeline = Pipeline([('preprocessing', preprocessing_df),
                    ('classifier', logistic)])

#split data into training and test sets
X=df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y=df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
# find the best parameters for the model using GridSearchCV
param_grid = {
    'logistic__penalty' : ['l1', 'l2'],
    'logistic__C' : [0.001, 0.01, 0.1, 1, 10, 100],
    'logistic__solver' : ['liblinear']}

# create gridsearch object
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# fit grid search
grid.fit(X_train, y_train)
'''
#fit the model
pipeline.fit(X_train, y_train)

# get accuracy score
print('Accuracy: ', accuracy_score(y_test, pipeline.predict(X_test)))


#store in pickle
pickle.dump(pipeline, open('model.pkl', 'wb'))