from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn import tree
import csv
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import scipy
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

def encoder(data):
    '''Map the categorical variables to numbers to work with scikit learn'''
    for col in data.columns:
        if data.dtypes[col] == "object":
            le = preprocessing.LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
    return data

def age_of_kids_test():
    global dfTest
    '''Look at the name title and find kids, then input mean value for kids to kids with nan age'''
    # Label kids
    dfTest["iskid"] = ["Yes" if re.search("Master",i) else "No" for i in dfTest["Name"]]
    # Find mean
    mean_age_kids = np.median(dfTest[dfTest["iskid"] == "Yes"]["Age"])
    # Find index of kids with na
    index_kids_nullage = dfTest[(dfTest["iskid"] == "Yes") & (dfTest["Age"].isnull())].index
    # Create a new vector of ages inputing mean age for kids to kids
    newage = dfTest["Age"].tolist()
    for i in index_kids_nullage:
        newage[i] = mean_age_kids
    # Assigning variable
    dfTest["Age"] = newage
    return dfTest.drop("iskid",1)

def fill_age_test(columns_needed):
    global dfTest
    '''Fill age by using Random Forest'''
        # Train and Test sets
    train_age_notnull = dfTest[~dfTest["Age"].isnull()]
    train_age_null = dfTest[dfTest["Age"].isnull()]
    # Trains set X and Y
    X_train_age_notnull = encoder(train_age_notnull[columns_needed])
    Y_train_age_notnull = train_age_notnull['Age']
    # Grid Search
    print('processing GridSearch')
    parameters = {"max_depth": [2,3,4,5,6,7,8,9,10,11,12]
                    ,"min_samples_split" :[2,3,4,5,6]
                    ,"n_estimators" : [10]
                    ,"min_samples_leaf": [1,2,3,4,5]
                    ,"max_features": (2,3,4)}
    rf_regr = RandomForestRegressor()
    age_model = GridSearchCV(rf_regr,parameters, n_jobs = 3, cv = 10)
    age_fit = age_model.fit(X_train_age_notnull,Y_train_age_notnull)
    learned_parameters = age_fit.best_params_
    # Rerun model on fitted parameters
    rfr_age = RandomForestRegressor(max_depth = learned_parameters["max_depth"]
                        ,max_features = learned_parameters['max_features']
                        ,min_samples_leaf = learned_parameters['min_samples_leaf']
                        ,min_samples_split = learned_parameters['min_samples_split']
                        ,n_estimators = 500
                        ,n_jobs = 3)
    print('Running Model')
    rfr_fit = rfr_age.fit(X_train_age_notnull,Y_train_age_notnull)
    print('Done')
    # Collate data together
    train_age_null['Age'] = rfr_fit.predict(encoder(train_age_null[columns_needed]))
    data = pd.concat([train_age_null,train_age_notnull])
    return data

def age_of_kids():
    global df
    '''Look at the name title and find kids, then input mean value for kids to kids with nan age'''
    # Label kids
    df["iskid"] = ["Yes" if re.search("Master",i) else "No" for i in df["Name"]]
    # Find mean
    mean_age_kids = np.median(df[df["iskid"] == "Yes"]["Age"])
    # Find index of kids with na
    index_kids_nullage = df[(df["iskid"] == "Yes") & (df["Age"].isnull())].index
    # Create a new vector of ages inputing mean age for kids to kids
    newage = df["Age"].tolist()
    for i in index_kids_nullage:
        newage[i] = mean_age_kids
    # Assigning variable
    df["Age"] = newage
    return df.drop("iskid",1)

def fill_age(columns_needed):
    global df
    '''Fill age by using Random Forest'''
        # Train and Test sets
    train_age_notnull = df[~df["Age"].isnull()]
    train_age_null = df[df["Age"].isnull()]
    # Trains set X and Y
    X_train_age_notnull = encoder(train_age_notnull[columns_needed])
    Y_train_age_notnull = train_age_notnull['Age']
    # Grid Search
    print('processing GridSearch')
    parameters = {"max_depth": [2,3,4,5,6,7,8,9,10,11,12]
                    ,"min_samples_split" :[2,3,4,5,6]
                    ,"n_estimators" : [10]
                    ,"min_samples_leaf": [1,2,3,4,5]
                    ,"max_features": (2,3,4)}
    rf_regr = RandomForestRegressor()
    age_model = GridSearchCV(rf_regr,parameters, n_jobs = 3, cv = 10)
    age_fit = age_model.fit(X_train_age_notnull,Y_train_age_notnull)
    learned_parameters = age_fit.best_params_
    # Rerun model on fitted parameters
    rfr_age = RandomForestRegressor(max_depth = learned_parameters["max_depth"]
                        ,max_features = learned_parameters['max_features']
                        ,min_samples_leaf = learned_parameters['min_samples_leaf']
                        ,min_samples_split = learned_parameters['min_samples_split']
                        ,n_estimators = 500
                        ,n_jobs = 3)
    print('Running Model')
    rfr_fit = rfr_age.fit(X_train_age_notnull,Y_train_age_notnull)
    print('Done')
    # Collate data together
    train_age_null['Age'] = rfr_fit.predict(encoder(train_age_null[columns_needed]))
    data = pd.concat([train_age_null,train_age_notnull])
    return data

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

def title_map(title):
    if title in ['Capt', 'Col', 'Major', 'Dr', 'Rev']:
        return 0
    elif title in ['Jonkheer','Don', 'Sir', 'the Countess', 'Dona', 'Lady']:
        return 1
    elif title in ['Mme', 'Ms', 'Mrs']:
        return 2
    elif title in ['Miss','Mlle']:
        return 3
    elif title in ['Master']:
        return 4
    elif title in ['Mr']:
        return 5
    else:
        return 6

pd.options.mode.chained_assignment = None

#load data
df = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

#Preprocessing
le = preprocessing.LabelEncoder()
df['Title'] = df['Name'].apply(get_title).apply(title_map)
df.Title = le.fit_transform(df.Title)
dfTest['Title'] = dfTest['Name'].apply(get_title).apply(title_map)
dfTest.Title = le.fit_transform(dfTest.Title)

#Sex
df.Sex = le.fit_transform(df.Sex)
dfTest.Sex = le.fit_transform(dfTest.Sex)

#Fare
df.Fare.fillna(df.Fare.median(),inplace=True)
df.loc[ df['Fare'] <= 17, 'Fare'] = 0
df.loc[(df['Fare'] > 17) & (df['Fare'] <= 30), 'Fare'] = 1
df.loc[(df['Fare'] > 30) & (df['Fare'] <= 100), 'Fare'] = 2
df.loc[ df['Fare'] > 100, 'Fare'] = 3
df['Fare'] = df['Fare'].astype(int)
dfTest.Fare.fillna(dfTest.Fare.median(),inplace=True)
dfTest.loc[ dfTest['Fare'] <= 17, 'Fare'] = 0
dfTest.loc[(dfTest['Fare'] > 17) & (dfTest['Fare'] <= 30), 'Fare'] = 1
dfTest.loc[(dfTest['Fare'] > 30) & (dfTest['Fare'] <= 100), 'Fare'] = 2
dfTest.loc[ dfTest['Fare'] > 100, 'Fare'] = 3
dfTest['Fare'] = dfTest['Fare'].astype(int)


#Embarked
df.Embarked.fillna("S",inplace = True)
df.Embarked = le.fit_transform(df.Embarked)
dfTest.Embarked.fillna("S",inplace = True)
dfTest.Embarked = le.fit_transform(dfTest.Embarked)

#Family
df['Family'] =  df["Parch"] + df["SibSp"] + 1
dfTest['Family'] =  dfTest["Parch"] + dfTest["SibSp"]

#Cabin
df['Cabin'] = df['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
df['Cabin'] = df['Cabin'].map(cabin_mapping)
df['Cabin'] = df.groupby(['Pclass'])['Cabin'].transform(lambda x: x.fillna(np.nanmedian(x)))
df['Cabin'] = df['Cabin'].astype(int)
dfTest['Cabin'] = dfTest['Cabin'].str[:1]
dfTest['Cabin'] = dfTest['Cabin'].map(cabin_mapping)
dfTest['Cabin'] = dfTest.groupby(['Pclass'])['Cabin'].transform(lambda x: x.fillna(np.nanmedian(x)))
dfTest['Cabin'] = dfTest['Cabin'].astype(int)


#Age
nullAges = age_of_kids()
nullAges = fill_age(["Title", "Pclass", "Sex", "Fare", "Family", "Cabin", "Embarked",])
df = nullAges.sort_values("PassengerId")
nullAgesTest = age_of_kids_test()
nullAgesTest = fill_age_test(["Title", "Pclass", "Sex", "Fare", "Family", "Cabin", "Embarked",])
dfTest = nullAgesTest.sort_values("PassengerId")

df['Age'] = df['Age'].astype(int)
df["IsChild"] = df['Age'] < 16
df["IsOld"] = df['Age'] > 63
df.IsChild = le.fit_transform(df.IsChild)
df.IsOld = le.fit_transform(df.IsOld)

dfTest['Age'] = dfTest['Age'].astype(int)
dfTest["IsChild"] = dfTest['Age'] < 16
dfTest["IsOld"] = dfTest['Age'] > 63
dfTest.IsChild = le.fit_transform(dfTest.IsChild)
dfTest.IsOld = le.fit_transform(dfTest.IsOld)

df.loc[df['Age'] < 16, 'Age'] = 0
df.loc[(df['Age'] >= 16) & (df['Age'] <= 63), 'Age'] = 1
df.loc[(df['Age'] > 63), 'Age'] = 2
df['age_class'] = df['Age'] * df['Pclass']
dfTest.loc[dfTest['Age'] < 16, 'Age'] = 0
dfTest.loc[(dfTest['Age'] >= 16) & (dfTest['Age'] <= 63), 'Age'] = 1
dfTest.loc[(dfTest['Age'] > 63), 'Age'] = 2
dfTest['age_class'] = dfTest['Age'] * dfTest['Pclass']


#Model
dfX = df.loc[:,["Title", "Pclass", "Sex", "Fare", "Family", "Cabin", "Embarked", "age_class", "IsChild", "IsOld", "Age"]]


X = dfX.values
y = df.loc[:, df.columns == 'Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
rfc = RandomForestClassifier()
param_grid = {
    "max_depth": [6,9, 12]
    ,"min_samples_split" :[6, 9]
    ,"n_estimators" : [200, 300]
    ,"min_samples_leaf": [6, 9]
    ,"max_features": (6,9,"sqrt")
    ,"criterion": ('gini','entropy')

}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train.ravel())
learned_parameters = CV_rfc.best_params_
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)
print("Trained....")
rfc = RandomForestClassifier(max_depth = learned_parameters["max_depth"]
                            ,max_features = learned_parameters['max_features']
                            ,min_samples_leaf = learned_parameters['min_samples_leaf']
                            ,min_samples_split = learned_parameters['min_samples_split']
                            ,criterion = learned_parameters['criterion']
                            ,n_estimators = 5000
                            ,n_jobs = 3)
CV_rfc.fit(X_train, y_train.ravel())

#Predict
y_pred = CV_rfc.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy1: %.2f%%" % (accuracy * 100.0))

#Predict Test
dfTestX = dfTest.loc[:,["Title", "Pclass", "Sex", "Fare", "Family", "Cabin", "Embarked", "age_class", "IsChild", "IsOld", "Age"]]
y_pred = CV_rfc.predict(dfTestX)

predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy2: %.2f%%" % (accuracy * 100.0))

submission = pd.DataFrame({
        "PassengerId": dfTest["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)


