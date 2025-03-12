import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Tiktok Sentiment Visualization",
    page_icon="📊",
    layout="wide",
)

st.title("Tiktok Caption Sentiment Analysis")
st.markdown("Project on Data Science (Predicts if a video is positive, negative, or neutral using NLTK Pre-trained Vader)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Nltk_Vader", "Sentiment_results.csv")

df = pd.read_csv(csv_path)

# Count the number of occurrences for each sentiment
sentiment_counts = df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment_label', 'count']

col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
    sentiment_counts,
    values='count',  # Use count instead of sentiment_score
    names='sentiment_label',
    title='Sentiment Distribution',
    color='sentiment_label',
    color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
)
st.plotly_chart(fig_pie, use_container_width=True)



with col2:
    fig_bar = px.bar(
        sentiment_counts,
        x='sentiment_label',
        y='count',  # Fixed issue: Use 'count' instead of 'sentiment_score'
        title='Sentiment Counts',
        color='sentiment_label',
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.header('Sentiment Scores Visualization')

col3, col4 = st.columns(2)

with col3:
    fig_scatter = px.scatter(
        df,
        x='positive',
        y='negative',
        color='sentiment_label',  # Fixed issue: Corrected column name
        hover_data=['cleaned_caption'] if 'cleaned_caption' in df.columns else None,
        title='Positive Vs Negative Sentiment Scores',
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    fig_3d = px.scatter_3d(
        df,
        x='positive',
        y='negative',
        z='neutral',
        color='sentiment_label',
        hover_data=['text'] if 'text' in df.columns else None,
        title='Positive vs Negative Sentiment Scores',
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    fig_3d.update_layout(scene=dict(
        xaxis_title='Positive',
        yaxis_title='Negative',
        zaxis_title='Neutral'
    ))
    st.plotly_chart(fig_3d, use_container_width=True)
