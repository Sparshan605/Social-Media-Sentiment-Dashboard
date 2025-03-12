import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.header('File head')
st.write(df.head())

# Count the number of occurrences for each sentiment
df['sentiment_label'] = df['sentiment_label'].str.lower().str.strip()
sentiment_counts = df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
sentiment_counts['sentiment_count'] = pd.to_numeric(sentiment_counts['sentiment_count'], errors='coerce')
st.header('Counts')
st.write(sentiment_counts)



col1, col2 = st.columns(2)

with col1:
    labels = sentiment_counts['sentiment_label'].tolist()
    values = sentiment_counts['sentiment_count'].tolist()

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=['gray', 'green', 'red'])
    )])

    fig_pie.update_layout(
        title='Sentiment Distribution',
        margin=dict(l=20, r=20, t=80, b=0),  
        height=400,  
        width=400,
        )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    labels = sentiment_counts['sentiment_label'].tolist()
    values = sentiment_counts['sentiment_count'].tolist()
    fig_bar = go.Figure(data=[
        go.Bar(
            x=labels,  
            y=values,  
            marker=dict(
            color=['gray' if x == 'neutral' else 'green' if x == 'positive' else 'red' 
                   for x in sentiment_counts['sentiment_label']]
        ),
            orientation='v',
        )
    ])

    # Updating layout for better appearance
    fig_bar.update_layout(
        xaxis_title='Sentiment Label',
        yaxis_title='Count',
        bargap=0.1, 
        barmode='group',
        font=dict(color='white')  
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.header('Sentiment Scores Visualization')

col3, col4 = st.columns(2)
with col3:
    fig_hist = px.histogram(df, x='sentiment_score', color='sentiment_label')
    st.plotly_chart(fig_hist, use_container_width=True)
# with col3:
#     fig_scatter = px.scatter(
#         df,
#         x='positive',
#         y='negative',
#         color='sentiment_label',  # Fixed issue: Corrected column name
#         hover_data=['cleaned_caption'] if 'cleaned_caption' in df.columns else []
#         title='Positive Vs Negative Sentiment Scores',
#         color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
#     )
#     st.plotly_chart(fig_scatter, use_container_width=True)

# with col4:
#     fig_3d = px.scatter_3d(
#         df,
#         x='positive',
#         y='negative',
#         z='neutral',
#         color='sentiment_label',
#         hover_data=['text'] if 'text' in df.columns else None,
#         title='Positive vs Negative Sentiment Scores',
#         color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
#     )
#     fig_3d.update_layout(scene=dict(
#         xaxis_title='Positive',
#         yaxis_title='Negative',
#         zaxis_title='Neutral'
#     ))
#     st.plotly_chart(fig_3d, use_container_width=True)