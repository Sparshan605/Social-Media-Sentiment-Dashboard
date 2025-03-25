import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from nltk.corpus import stopwords
from prediction import predict

st.set_page_config(
    page_title="Tiktok Sentiment Visualization",
    page_icon="📊",
    layout="wide",
)

st.title("TikTok Caption Sentiment Analysis")
st.markdown("Project on Data Science (Predicts sentiment using NLTK Vader and Machine Learning Model)")

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# Sentiment analysis functions
def get_sentiment_scores(text):
    sia = SentimentIntensityAnalyzer()
    if isinstance(text, str):
        scores = sia.polarity_scores(text)
        return scores
    else:
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to process data
def process_data(df):
    # Check if dataframe has the required column
    text_column = None
    possible_text_columns = ['text', 'caption', 'comment_text', 'content']
    
    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        st.error("Error: The uploaded file must contain a text column (like 'text', 'caption', or 'comment_text')")
        return None
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Clean the text
    processed_df['cleaned_text'] = processed_df[text_column].apply(preprocess_text)
    
    # Apply NLTK VADER sentiment analysis
    processed_df['sentiment_scores'] = processed_df['cleaned_text'].apply(get_sentiment_scores)
    processed_df['negative'] = processed_df['sentiment_scores'].apply(lambda x: x['neg'])
    processed_df['neutral'] = processed_df['sentiment_scores'].apply(lambda x: x['neu'])
    processed_df['positive'] = processed_df['sentiment_scores'].apply(lambda x: x['pos'])
    processed_df['compound'] = processed_df['sentiment_scores'].apply(lambda x: x['compound'])
    processed_df['nltk_sentiment'] = processed_df['compound'].apply(get_sentiment_label)
    
    # Apply additional model prediction
    try:
        # Prepare text for vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(processed_df['cleaned_text'])
        
        # Get predictions from the additional model
        processed_df['model_sentiment'] = predict(processed_df['cleaned_text'])
    except Exception as e:
        st.warning(f"Error in model prediction: {str(e)}")
        processed_df['model_sentiment'] = 'Unable to predict'
    
    return processed_df

# Tabs for different views
tab1, tab2 = st.tabs(["Pre-loaded Data", "Upload New Data"])

with tab1:
    # Try to load pre-existing data if available
    try:
        df = pd.read_csv("C:/Users/Hp/Documents/Projects/Social Media Sentiment Dashboard/Nltk_Vader/Sentiment_results.csv")
        st.header('Pre-loaded Data')
        st.write(df.head())
        
        # Count the number of occurrences for each sentiment
        df['nltk_sentiment'] = df['nltk_sentiment'].str.lower().str.strip()
        sentiment_counts = df['nltk_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
        sentiment_counts['sentiment_count'] = pd.to_numeric(sentiment_counts['sentiment_count'], errors='coerce')
        
        st.header('Sentiment Counts')
        st.write(sentiment_counts)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            labels = sentiment_counts['sentiment_label'].tolist()
            values = sentiment_counts['sentiment_count'].tolist()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=['gray' if x == 'neutral' else 'green' if x == 'positive' else 'red' 
                               for x in sentiment_counts['sentiment_label']])
            )])
            
            fig_pie.update_layout(
                title='Sentiment Distribution',
                margin=dict(l=20, r=20, t=80, b=0),
                height=400,
                width=400,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts['sentiment_label'],
                    y=sentiment_counts['sentiment_count'],
                    marker=dict(
                        color=['gray' if x == 'neutral' else 'green' if x == 'positive' else 'red' 
                               for x in sentiment_counts['sentiment_label']]
                    ),
                    orientation='v',
                )
            ])
            
            fig_bar.update_layout(
                title='Sentiment Count by Category',
                xaxis_title='Sentiment Label',
                yaxis_title='Count',
                bargap=0.1,
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Couldn't load pre-existing data: {str(e)}")
        st.info("You can upload your own data in the 'Upload New Data' tab.")

with tab2:
    st.header("Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload your TikTok comments/captions data (CSV file)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            with st.expander("View raw data sample"):
                st.dataframe(data.head())
            
            with st.spinner('Processing sentiment analysis...'):
                processed_data = process_data(data)
            
            if processed_data is not None:
                st.success('Sentiment analysis complete!')
                
                # Display processed data
                st.subheader("Processed Data Preview")
                st.dataframe(processed_data.head())
                
                # NLTK Sentiment Counts
                nltk_sentiment_counts = processed_data['nltk_sentiment'].value_counts().reset_index()
                nltk_sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
                
                st.subheader('NLTK Sentiment Counts')
                st.write(nltk_sentiment_counts)
                
                if 'model_sentiment' in processed_data.columns:
                    model_sentiment_counts = processed_data['model_sentiment'].value_counts().reset_index()
                    model_sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
                    
                    st.subheader('Model Sentiment Counts')
                    st.write(model_sentiment_counts)
                
                # Sentiment Comparison
                if 'model_sentiment' in processed_data.columns:
                    st.subheader('Sentiment Comparison')
                    comparison_df = processed_data.groupby(['nltk_sentiment', 'model_sentiment']).size().reset_index(name='count')
                    pivot_table = pd.pivot_table(
                        comparison_df, 
                        values='count', 
                        index='nltk_sentiment', 
                        columns='model_sentiment', 
                        fill_value=0
                    )
                    st.write("Sentiment Comparison Matrix:")
                    st.write(pivot_table)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    labels = nltk_sentiment_counts['sentiment_label'].tolist()
                    values = nltk_sentiment_counts['sentiment_count'].tolist()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=['gray' if x == 'neutral' else 'green' if x == 'positive' else 'red' 
                                       for x in nltk_sentiment_counts['sentiment_label']])
                    )])
                    
                    fig_pie.update_layout(
                        title='NLTK Sentiment Distribution',
                        margin=dict(l=20, r=20, t=80, b=0),
                        height=400,
                        width=400,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    nltk_sentiment_counts = processed_data['nltk_sentiment'].value_counts().reset_index()
                    nltk_sentiment_counts.columns = ['sentiment_label', 'sentiment_count']

                    # Ensure lowercase and stripped labels
                    nltk_sentiment_counts['sentiment_label'] = nltk_sentiment_counts['sentiment_label'].str.lower().str.strip()

                    # Color mapping
                    color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}

                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=nltk_sentiment_counts['sentiment_label'],
                            y=nltk_sentiment_counts['sentiment_count'],
                            marker=dict(
                                color=[color_map.get(x, 'black') for x in nltk_sentiment_counts['sentiment_label']]
                            ),
                            orientation='v',
                        )
                    ])

                    fig_bar.update_layout(
                        title='NLTK Sentiment Count by Category',
                        xaxis_title='Sentiment Label',
                        yaxis_title='Count',
                        bargap=0.1,
                        barmode='group'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Print out the data for verification
                    st.write("Sentiment Counts:", nltk_sentiment_counts)
                st.subheader("Sentiment Score Distribution")
                fig_scores = go.Figure()
                fig_scores.add_trace(go.Box(y=processed_data['negative'], name='Negative'))
                fig_scores.add_trace(go.Box(y=processed_data['neutral'], name='Neutral'))
                fig_scores.add_trace(go.Box(y=processed_data['positive'], name='Positive'))
                fig_scores.add_trace(go.Box(y=processed_data['compound'], name='Compound'))
                
                fig_scores.update_layout(
                    title='Distribution of Sentiment Scores',
                    yaxis_title='Score Value',
                    boxmode='group'
                )
                st.plotly_chart(fig_scores, use_container_width=True)
                csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="tiktok_sentiment_analysis_results.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")