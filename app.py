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
    fig_pie =px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Sentiment Distribution',
        color='Sentiment_label',
        color_discrete_map={'positive':'green','neutral':'gray','negative':'red'})
    st.plotly_chart(fig_pie,use_container_width=True)

with col2:
    fig_bar=px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        title='Sentiment Counts',
        color='sentiment_label',
        color_discrete_map={'positive':'green','neutral':'gray','negative':'red'} 
    )
    st.plotly_chart(fig_bar,use_container_width=True)

st.header('sentiment scores visualization')
col3, col4= st.columns(2)

with col3:
    fig_scatter=px.scatter(
        df,
        x='positive',
        y='negative',
        color='setiment_label',
        hover_data=['cleaned_caption'] if 'cleaned_caption' in df.columns else None,
        title='Positive Vs Negative Sentiment Scores',
        color_discrete_map={'positive':'green','neutral':'gray','negative':'red'}
    )
    st.plotly_chart(fig_scatter,use_container_width=True)

with col4:
    fig_3d =px.scatter_3d(
        df,
        x='positive',
        y='negative',
        z='neutral',
        color='sentiment_label',
        hover_data=['text'] if 'text' in df.columns else None,
        title='Positive vs Negative Sentiment Scores',
        color_discrete_map={'positive':'green','neutral':'gray','negative':'red'}
    )
    fig_3d.update_layout(scene=dict(
        xaxis_title='Positive',
        yaxis_title='Negative',
        zaxis_title='Neutral'
    ))
    st.plotly_chart(fig_3d, use_container_width=True)




         
