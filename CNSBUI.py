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
import Analyticsandencoding as ane
import classify as cl


html_temp = """
    <div style="background-color:black ;padding:10px">
    <h1 style="color:white;text-align:center;">Market Guru</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)






st.sidebar.markdown('<b>ABOUT:</b>', unsafe_allow_html=True)
st.sidebar.markdown("Market Guru is an end to end Customer Relationship Manager that enables the decision makers at your organization to formulate precise customer retention strategies thus empowering your organization to save resources spent on customer retention campaigns and employ a focused marketing approach. The entire process is divided into 2 stages: stage1 and stage2.")
st.sidebar.markdown('<b>Stage 1:</b>', unsafe_allow_html=True)
st.sidebar.markdown("The end user enters the customer data of the organization.Based on the entered customer data of the organization, Market guru(stage1) zeros down on the factors that are impacting customer churn at your organization the most and provides you with the analytics for the same, thus enabling you to formulate an articulate customer retention strategy for your organization based on the analysis.")

st.sidebar.markdown('<b>How does Market Guru empower you?:</b>', unsafe_allow_html=True)  
st.sidebar.markdown("1.Prima facie analysis shows that Market Guru enables you to save 76% of time, energy and resources spent on customer retention campaigns.")
st.sidebar.markdown("2.Enables you to articulately plan your customer retention campaign by zeroing in on the most statistically significant factors that are impacting the customer churn at the organization.") 
st.sidebar.markdown("3.Automates the end to end process of Analytics and model buiding and hence eliminates the need for a Third Party Analyst; thus, makes your organization Atmanirbhar!")
st.sidebar.markdown('<b>Built by:</b>', unsafe_allow_html=True)
st.sidebar.markdown('Team : Hackers101')


html_temp = """
    <div style="background-color:white ;padding:10px">
    <h3 style="color:black;text-align:center;">Enter your customer data. Market Guru will analyse it and present to you a Customer Churn analytics report based on your data.</h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)





uploaded_file = st.file_uploader("Please enter your Customer Data here:", type=["csv"]) 
def upload():
    df1=pd.read_csv(uploaded_file)
    return(df1)
    

if st.button("Analyse"):
        df1=upload()
        ane.Analyzeandencode(df1)
        cl.ml(df1)
        html_temp69 = """
        <div style="background-color:white ;padding:10px">
        <h3 style="color:black;text-align:left;">A model has been successfully built around your data. you can proceed to the next stage here:</h3>
        </div>
        """
        st.markdown(html_temp69, unsafe_allow_html=True)
        st.write('https://marketguru2.herokuapp.com/')
        
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        