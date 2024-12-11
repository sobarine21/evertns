import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time
import re

# Streamlit app configuration
st.set_page_config(page_title="Customer Support Analysis", layout="wide")
st.title("Customer Support Transcript Analyzer")
st.markdown("""
    Upload the transcript to analyze agent performance, sentiment, empathy, and much more. Get insights and actionable feedback!
""")

# File upload
uploaded_file = st.file_uploader("Upload Transcript", type=["txt", "pdf"])

# Helper functions
def parse_transcript(transcript_text):
    """Parse the transcript text into a list of tuples (speaker, text)."""
    conversations = []
    speaker = None
    for line in transcript_text.split("\n"):
        if "Agent" in line:
            speaker = "Agent"
        elif "Customer" in line:
            speaker = "Customer"
        else:
            if speaker:
                conversations.append((speaker, line.strip()))
    return conversations

def check_profanity(text):
    """Check if any profanity is present in the text."""
    profanities = ["bsdk", "badword1", "badword2"]  # Add more profanities here
    for word in profanities:
        if word in text.lower():
            return True
    return False

def sentiment_analysis(text):
    """Analyze sentiment polarity using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

def emotion_detection(text):
    """Detect emotions based on specific keywords."""
    positive_keywords = ["thanks", "appreciate", "grateful", "happy"]
    negative_keywords = ["angry", "frustrated", "upset", "disappointed"]
    if any(word in text.lower() for word in positive_keywords):
        return "Positive Emotion"
    elif any(word in text.lower() for word in negative_keywords):
        return "Negative Emotion"
    return "Neutral Emotion"

def topic_modeling(text_data):
    """Use LDA to extract topics from the conversation."""
    vectorizer = CountVectorizer(stop_words="english")
    data_vectorized = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    lda.fit(data_vectorized)
    topics = lda.components_
    return topics

def key_phrase_extraction(text):
    """Extract key phrases using simple regex (could use more advanced NLP tools)."""
    key_phrases = re.findall(r'\b(?:issue|problem|help|solution|assist)\b', text, re.IGNORECASE)
    return key_phrases

def generate_report(conversations):
    """Generate a detailed report analyzing the conversation."""
    empathy = False
    assistance = False
    for speaker, text in conversations:
        if speaker == "Agent":
            if "sorry" in text or "understand" in text:
                empathy = True
            if "help" in text or "assist" in text:
                assistance = True
    return empathy, assistance

def process_transcript(transcript_text):
    """Process the transcript and provide a detailed analysis."""
    conversations = parse_transcript(transcript_text)
    empathy, assistance = generate_report(conversations)
    
    sentiment = [sentiment_analysis(text) for _, text in conversations]
    emotions = [emotion_detection(text) for _, text in conversations]
    key_phrases = [key_phrase_extraction(text) for _, text in conversations]

    topics = topic_modeling([text for _, text in conversations])

    return {
        "conversations": conversations,
        "sentiment": sentiment,
        "emotions": emotions,
        "key_phrases": key_phrases,
        "topics": topics,
        "empathy": empathy,
        "assistance": assistance,
    }

# Display analysis options
if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode("utf-8") if uploaded_file.type == "text/plain" else "PDF file detected, please upload a text file for analysis."

    st.write("### Transcript Preview")
    st.text_area("Transcript", transcript_text, height=200)

    if st.button("Start Analysis"):
        with st.spinner("Analyzing... This may take a few seconds..."):
            analysis_result = process_transcript(transcript_text)
        
        # Display results
        st.write("### Sentiment Analysis")
        sentiment_count = pd.Series(analysis_result["sentiment"]).value_counts()
        st.bar_chart(sentiment_count)

        st.write("### Emotional Tone Detection")
        emotion_count = pd.Series(analysis_result["emotions"]).value_counts()
        st.bar_chart(emotion_count)

        st.write("### Key Phrases Detected")
        st.write(analysis_result["key_phrases"])

        st.write("### Empathy and Assistance Evaluation")
        if analysis_result["empathy"]:
            st.write("The agent showed empathy during the conversation.")
        else:
            st.write("The agent did not show clear empathy.")

        if analysis_result["assistance"]:
            st.write("The agent provided assistance to the customer.")
        else:
            st.write("The agent did not provide clear assistance.")

        # Visualizing topics with a word cloud
        st.write("### Topic Modeling (Top Keywords)")
        topics = analysis_result["topics"]
        for topic_idx, topic in enumerate(topics):
            st.write(f"Topic #{topic_idx + 1}:")
            words = [f"{word}" for word in topic.argsort()[:-11:-1]]
            st.write(" ".join(words))
        
        # Allowing download of the analysis results
        if st.button("Download Report"):
            results_df = pd.DataFrame(analysis_result)
            results_df.to_csv("conversation_analysis_report.csv", index=False)
            st.success("Report has been saved as 'conversation_analysis_report.csv'")
