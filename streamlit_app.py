import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from deepmoji import DeepMoji
from deepmoji.model_def import deepmoji_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# Setup
nlp = spacy.load("en_core_web_sm")  # Load spaCy model
analyzer = SentimentIntensityAnalyzer()  # VADER sentiment analyzer
deepmoji_model = deepmoji_model.load_model()  # Load DeepMoji model

# Streamlit App
st.title("Customer Support Transcript Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload CSV with customer support transcripts", type=["csv"])

if uploaded_file:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Transcript Data", df.head())

    # Sentiment Analysis Function using VADER
    def analyze_sentiment(text):
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']

    # Named Entity Recognition (NER) with SpaCy
    def extract_entities(text):
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return entities

    # Emotion Detection with DeepMoji
    def predict_emotion(text):
        tokenizer = DeepMojiTokenizer()
        tokens = tokenizer.tokenize(text)
        padded_tokens = pad_sequences([tokens], maxlen=30, truncating='post')
        prediction = deepmoji_model.predict(padded_tokens)
        return prediction.argmax(axis=1)[0]  # Emotion label

    # Topic Modeling using LDA
    def extract_topics(documents, n_topics=3):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(documents)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_topics = lda.fit_transform(X)

        terms = vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(lda.components_):
            topics.append([terms[i] for i in topic.argsort()[:-6 - 1:-1]])
        return topics

    # Process each transcript
    sentiments = []
    emotions = []
    topics = []

    for text in df['transcript']:  # Assuming the column containing transcripts is named 'transcript'
        sentiments.append(analyze_sentiment(text))
        emotions.append(predict_emotion(text))
        entities = extract_entities(text)
        topics.append(entities)

    # Add sentiment, emotion, and topics to DataFrame
    df['sentiment'] = sentiments
    df['emotion'] = emotions
    df['topics'] = topics

    # Display the DataFrame with sentiment and emotion labels
    st.write("Transcript Analysis with Sentiment, Emotion, and Topics", df)

    # Visualization: Sentiment Distribution
    def plot_sentiment_distribution(sentiments):
        plt.figure(figsize=(10, 6))
        plt.hist(sentiments, bins=20, color='skyblue', edgecolor='black')
        plt.title("Sentiment Distribution of Conversations")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        st.pyplot()

    plot_sentiment_distribution(sentiments)

    # Visualization: Word Cloud
    def generate_wordcloud(texts):
        text = " ".join(texts)
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

    generate_wordcloud(df['transcript'])

    # Visualization: Topic Modeling
    def plot_topic_modeling(topics):
        topic_text = "\n".join([f"Topic {i+1}: " + " ".join(topic) for i, topic in enumerate(topics)])
        st.text(topic_text)

    plot_topic_modeling(extract_topics(df['transcript']))

    # Save output as CSV
    output_file = "transcript_analysis.csv"
    df.to_csv(output_file, index=False)

    # Provide download link for the output CSV
    st.download_button(
        label="Download Transcript Analysis CSV",
        data=df.to_csv(index=False),
        file_name=output_file,
        mime="text/csv"
    )

    # Save and download visualization images
    def save_and_download_plot():
        # Sentiment Distribution Plot
        plt.figure(figsize=(10, 6))
        plt.hist(sentiments, bins=20, color='skyblue', edgecolor='black')
        plt.title("Sentiment Distribution of Conversations")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        sentiment_plot_path = "/mnt/data/sentiment_distribution.png"
        plt.savefig(sentiment_plot_path)
        st.download_button(
            label="Download Sentiment Distribution Plot",
            data=open(sentiment_plot_path, "rb").read(),
            file_name="sentiment_distribution.png",
            mime="image/png"
        )

        # Word Cloud Plot
        wordcloud = WordCloud(width=800, height=400).generate(" ".join(df['transcript']))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        wordcloud_plot_path = "/mnt/data/wordcloud.png"
        plt.savefig(wordcloud_plot_path)
        st.download_button(
            label="Download Word Cloud Plot",
            data=open(wordcloud_plot_path, "rb").read(),
            file_name="wordcloud.png",
            mime="image/png"
        )

    save_and_download_plot()
