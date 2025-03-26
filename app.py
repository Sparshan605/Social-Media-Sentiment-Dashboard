import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.warning(f"Error downloading NLTK resources: {e}")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Import prediction function (ensure this is correctly implemented)
try:
    from prediction import predict
except ImportError:
    st.error("Could not import prediction module. Prediction functionality will be disabled.")
    predict = None

# Configure Streamlit page
st.set_page_config(
    page_title="TikTok Sentiment Visualization",
    page_icon="📊",
    layout="wide",
)

# Text preprocessing function
def preprocess_text(text):
    """Preprocess input text by cleaning and removing stopwords"""
    try:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])
        
        return text
    except Exception as e:
        st.warning(f"Error in text preprocessing: {e}")
        return ""

# Sentiment analysis functions
def get_sentiment_scores(text):
    """Calculate sentiment scores using NLTK VADER"""
    sia = SentimentIntensityAnalyzer()
    
    if isinstance(text, str):
        try:
            scores = sia.polarity_scores(text)
            return scores
        except Exception as e:
            st.warning(f"Error in sentiment scoring: {e}")
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    else:
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

def get_sentiment_label(compound_score):
    """Classify sentiment based on compound score"""
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to process data
def process_data(df):
    """Process input dataframe for sentiment analysis"""
    # Identify text column
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
        if predict is not None:
            # Prepare text for vectorization
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(processed_df['cleaned_text'])
            
            # Get predictions from the additional model
            processed_df['model_sentiment'] = predict(processed_df['cleaned_text'])
        else:
            processed_df['model_sentiment'] = 'Unable to predict'
    except Exception as e:
        st.warning(f"Error in model prediction: {str(e)}")
        processed_df['model_sentiment'] = 'Unable to predict'
    
    return processed_df

# Main Streamlit App
def main():
    st.title("TikTok Caption Sentiment Analysis")
    st.markdown("Project on Data Science (Predicts sentiment using NLTK Vader and Trained Logistic Regressor)")

    # Tabs for different views
    tab1, tab2 = st.tabs(["Pre-loaded Data", "Upload New Data"])

    with tab1:
        st.header('Pre-loaded Data')
        # Try to load pre-existing data
        try:
            # Use the same processing function for pre-loaded data
            df = pd.read_csv("Nltk_Vader/Sentiment_results.csv")
            
            # Process the pre-loaded data using the same function
            processed_df = process_data(df)
            
            if processed_df is not None:
                # Display data and visualizations
                st.write(processed_df.head())
                
                # Sentiment counts
                sentiment_counts = processed_df['nltk_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
                st.header('Sentiment Counts')
                st.write(sentiment_counts)
                
                # Visualizations (similar to upload tab)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
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
                    # Bar chart
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
                    
                    # Rest of the visualization code remains the same as in the original script
                    # (Previous implementation for upload tab visualizations)
                    
                    # Display processed data
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_data.head())
                    
                    # NLTK Sentiment Counts
                    nltk_sentiment_counts = processed_data['nltk_sentiment'].value_counts().reset_index()
                    nltk_sentiment_counts.columns = ['sentiment_label', 'sentiment_count']
                    
                    st.subheader('NLTK Sentiment Counts')
                    st.write(nltk_sentiment_counts)
                    
                    # Visualization code (similar to the previous implementation)
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
                        # Bar chart and other visualizations remain the same
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=labels,
                                y=values,
                                marker=dict(
                                    color=['gray' if x == 'neutral' else 'green' if x == 'positive' else 'red' 
                                           for x in nltk_sentiment_counts['sentiment_label']]
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
                    
                    # Additional visualizations and download button
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()