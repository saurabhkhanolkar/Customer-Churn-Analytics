import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.combine import SMOTETomek


def ml(dataset):
    data_train=dataset.loc[dataset['years for cust']>10]
    data_train.drop(['Years since opening the account','years for cust'],axis=1,inplace=True)
    X = data_train.iloc[:, :-1].values
    y = data_train.iloc[:, -1].values
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    X[:,1] = le.fit_transform(X[:,1])
    X[:,2] = le.fit_transform(X[:,2])
    X[:,3] = le.fit_transform(X[:,3])
    X[:,4] = le.fit_transform(X[:,4])
    X[:,5] = le.fit_transform(X[:,5])
    X[:,6] = le.fit_transform(X[:,6])
    y = le.fit_transform(y)
    smk=SMOTETomek(random_state=42)
    Xres,yres=smk.fit_sample(X,y)
    #X_train, X_test, y_train, y_test = train_test_split(Xres, yres, test_size = 0.1, random_state = 18)
    X_train=Xres
    y_train=yres
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    import pickle
    pickle.dump(classifier,open('clss.pkl','wb'))


    daka=pd.read_csv('stage2data(1).csv')
    data_test=dataset.loc[dataset['years for cust']<=10]
    data_test.drop(['Years since opening the account','years for cust'],axis=1,inplace=True)
    X = data_test.iloc[:, :-1].values
    y = data_test.iloc[:, -1].values
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    X[:,1] = le.fit_transform(X[:,1])
    X[:,2] = le.fit_transform(X[:,2])
    X[:,3] = le.fit_transform(X[:,3])
    X[:,4] = le.fit_transform(X[:,4])
    X[:,5] = le.fit_transform(X[:,5])
    X[:,6] = le.fit_transform(X[:,6])
    y = le.fit_transform(y)
    X_test=X
    y_test=y        
    y_pred = classifier.predict(X_test)
    return(y_pred)
    for i in range(len(y_pred)):
        if(y_pred[i]==1):
            st.write('Custmer Number',daka['CustNo'][i],'using account number',daka['PrdAcctId'][i],'is at a high risk of churning out.' )
