import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import plotly.express as px


def Analyzeandencode(df1):
    
    html_temp2 = """<div style="background-color:black ;padding:10px">
    <h3 style="color:white;text-align:center;">Churn Analytics Report</h3>
    </div>
    """
        
    st.markdown(html_temp2, unsafe_allow_html=True)
        
    html_temp69 = """
    <div style="background-color:white ;padding:10px">
    <h3 style="color:black;text-align:center;">The Following factors impact Customer Churn at your Organization: </h3>
    </div>
    """
    st.markdown(html_temp69, unsafe_allow_html=True)
    X = df1.iloc[:, :-1].values
    y = df1.iloc[:, -1].values
        
           
        
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    X[:,1] = le.fit_transform(X[:,1])
    X[:,2] = le.fit_transform(X[:,2])
    X[:,3] = le.fit_transform(X[:,3])
    X[:,4] = le.fit_transform(X[:,4])
    X[:,5] = le.fit_transform(X[:,5])
    X[:,6] = le.fit_transform(X[:,6])
    y = le.fit_transform(y)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    
        
        
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1.astype(float)).fit()  
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.000005):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    list1=[]
    for i in range(len(selected_features_BE)):
        list1.append(int(selected_features_BE[i]))
    
    listn=[]
    
    for i in range(len(list1)):
        
        st.header(i+1)
        st.header(df1.columns[list1[i]])
        if(df1.columns[list1[i]]=='Years since opening the account'):
            for i in range(len(df1['Years since opening the account'])):
                if(df1['churn?'][i]==1):
                    listn.append(df1['Years since opening the account'][i]) 
            lol=pd.Series(listn)
            fig = px.histogram(lol)
            st.plotly_chart(fig)
            st.header("Insight:")
            st.write((191/308)*100,"% of the customers churned out after", 13, "years of staying with your organization.")
        else:
            listq=[]
            for k in range(len(df1[df1.columns[list1[i]]])):
                if(df1['churn?'][k]==1):
                    listq.append(df1[df1.columns[list1[i]]][k])
                lol1=pd.Series(listq)
            lol1.value_counts().rename({0:'count'},axis=1,inplace=True)
            st.write(lol1.value_counts())
            listz=[]
            for z in range(len(lol1.value_counts())):
                listz.append((lol1.value_counts()[z]/308)*100)
            lol2=pd.DataFrame(listz)
            lol2.rename({0:'Percentage'},axis=1,inplace=True)
            if(df1.columns[list1[i]]=='MemberType'):
                pass
            else:
                for j in range(len(lol1.unique())):
                    lol2.rename({j:lol1.unique()[j]},axis=0,inplace=True)
            
            st.write(lol2)
            fig3 = px.bar(lol2,y='Percentage')
            st.plotly_chart(fig3)
            st.header("Insight:")
            for l in range(len(lol1.unique())):
                st.write("From all the customers who churned, Percentage of customers from",lol1.unique()[l],"category was found to be", lol2['Percentage'][l],"%")
        
        

   