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
    fig = go.Figure()

# Add the box plot
    fig.add_trace(go.Box(
        y=df['sentiment_score'],
        name='Sentiment Scores',
        boxmean=True,
        boxpoints='all',  # Show all points
        jitter=0.3,
        pointpos=-1.8,
        line_color='rgb(8,81,156)',
        fillcolor='rgba(107,174,214,0.3)',
        marker=dict(
            color='rgb(107,174,214)',
            size=4
        )
    ))

    # Update layout
    fig.update_layout(
        title="Sentiment Score Box Plot",
        yaxis_title="Sentiment Score",
        height=400,
        width=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(
            gridcolor='white',
            zerolinecolor='red',  # Highlight the zero line
            range=[-1, 1]  # Keep your -0.9 to 0.9 range with padding
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=False)

    # Add some space (gap)
    st.write("")
    st.write("")

    # Display summary statistics with proper handling of NA values
    stats = {
        'Mean': np.mean(df['sentiment_score']).round(4),
        'Median': np.median(df['sentiment_score']).round(4),
        'Min': np.min(df['sentiment_score']).round(4),
        'Max': np.max(df['sentiment_score']).round(4),
        'Standard Deviation': np.std(df['sentiment_score']).round(4)
    }

    # Convert to DataFrame for display
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    st.write("Sentiment Score Statistics:")
    st.table(stats_df)


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