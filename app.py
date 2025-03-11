import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
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
  
col1, col2 = st.columns(2)

with col1:
    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_label','count']
    fig_pie =px.pie


         
