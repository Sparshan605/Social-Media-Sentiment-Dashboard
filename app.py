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
    st.write("### Alternative Visualization: Violin Plot")

    data = df['sentiment_score']

    # Add more pronounced jitter when values are mostly 0
    jitter_amount = 0.05  # Increased from 0.02
    jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(data))
    data_jittered = data + jitter

    fig_violin = go.Figure()

    fig_violin.add_trace(go.Violin(
        y=data_jittered,
        box_visible=True,
        meanline_visible=True,
        line_color='rgb(8,81,156)',
        fillcolor='rgba(107,174,214,0.6)',  # Increased opacity
        points='all',  # Show all points
        marker=dict(size=3, color='rgb(8,81,156)', opacity=0.5),
        bandwidth=0.15,  # Adjusted for better smoothing
        side='both',  # Create a symmetric violin
        spanmode='hard'  # Use the actual data range
    ))

    # Update layout
    fig_violin.update_layout(
        title="Sentiment Score Distribution (Violin Plot)",
        height=500,  # Increased height
        width=700,   # Increased width
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(
            title='Sentiment Score',
            gridcolor='white',
            range=[-1, 1],
            zerolinecolor='red',
            zerolinewidth=2  # Make zero line more prominent
        )
    )

    # Add annotation to highlight concentration at zero
    fig_violin.add_annotation(
        x=0,
        y=0,
        text="High concentration at 0",
        showarrow=True,
        arrowhead=1,
        ax=70,
        ay=-30,
        font=dict(size=12)
    )

    # Display in Streamlit
    st.plotly_chart(fig_violin, use_container_width=True)

# # with col3:
# #     fig_scatter = px.scatter(
# #         df,
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