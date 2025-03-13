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

col3, = st.columns(1)
with col3:
    df_sample = df.sample(min(200, len(df)), random_state=42)  # Reduce sample size
    df_sample['positive'] = df_sample['sentiment_score'].apply(lambda x: x if x > 0 else 0)
    df_sample['negative'] = df_sample['sentiment_score'].apply(lambda x: abs(x) if x < 0 else 0)
    df_sample['neutral'] = df_sample['sentiment_score'].apply(lambda x: 1 if x == 0 else 0)
    
    # Create 3D scatter plot with sampled data
    fig_3d = px.scatter_3d(
        df_sample,
        x='positive',
        y='negative',
        z='neutral',
        color='sentiment_label',
        title='Positive vs Negative Sentiment Scores',
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    )
    
    # Reduce marker size for better performance
    fig_3d.update_traces(marker=dict(size=3))  # Adjust marker size
    fig_3d.update_layout(scene=dict(
        xaxis_title='Positive Score',
        yaxis_title='Negative Score',
        zaxis_title='Neutral Score'
    ))

# Add error handling for Streamlit
try:
    st.plotly_chart(fig_3d, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying chart: {e}")
    st.write("Try reducing the sample size further if the issue persists.")
    
    # Add error handling
    try:
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying chart: {e}")
        st.write("Try reducing the sample size further if the issue persists.")
