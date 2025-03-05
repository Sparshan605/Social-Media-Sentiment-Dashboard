import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title= "Tiktok Sentiment Visualisation",
    page_icon="📊",
    layout= "wide" ,
)

st.title("Tiktok Caption Sentiment Analysis")
st.model('Project on Data science \
        (Predicts video is positive negative or neutral \
        (By using NLTK Pre trained Vader')

df=pd.read_csv('C:/Users/Hp/Documents/Projects/Social Media Sentiment Dashboard/Nltk_Vader/Sentiment_results.csv')

with st.expander('View raw data example'):
    st.dataframe(df.head())    
required_cols=['sentiment_label']

         
