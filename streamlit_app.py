import streamlit as st
import pandas as pd
from textblob import TextBlob

# Streamlit App UI
st.title("Customer Support Transcript Analyzer")
st.write("Analyze call transcripts for professionalism, empathy, tone, and sentiment.")

# File Upload
uploaded_file = st.file_uploader("Upload Call Transcript", type=["txt"])

# Function to parse and clean transcript
def parse_transcript(transcript_text):
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

# Function to check for profanity
def check_profanity(text):
    profanities = ["bsdk", "badword1", "badword2"]  # Add other bad words here
    for word in profanities:
        if word in text.lower():
            return True
    return False

# Function to analyze tone using TextBlob
def analyze_tone(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze conversation context
def analyze_conversation(text, speaker):
    if speaker == "Agent":
        if "apologize" in text or "understand" in text:
            return "Empathy"
        elif "help" in text or "resolve" in text:
            return "Resolution"
    return "General"

# Display and process transcript if uploaded
if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode("utf-8")
    conversations = parse_transcript(transcript_text)
    
    # Display the conversations
    for speaker, text in conversations:
        st.write(f"{speaker}: {text}")
    
    # Button to analyze
    if st.button("Analyze Transcript"):
        # Initialize the analysis results
        metrics = {"Empathy": [], "Resolution": [], "Tone": [], "Profanity": [], "Topic": []}
        
        # Loop through each conversation pair and analyze
        for speaker, text in conversations:
            try:
                # Check for profanity
                profanity = check_profanity(text)
                tone = analyze_tone(text)
                topic = analyze_conversation(text, speaker)
                
                # Update metrics
                metrics["Empathy"].append("Yes" if "Empathy" in topic else "No")
                metrics["Resolution"].append("Yes" if "Resolution" in topic else "No")
                metrics["Tone"].append(tone)
                metrics["Profanity"].append("Yes" if profanity else "No")
                metrics["Topic"].append(topic)
            
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Show the results in a table
        df = pd.DataFrame(metrics)
        st.write(df)
        
        # Display charts for analysis
        st.write("Metrics Overview")
        fig, ax = plt.subplots()
        df.count().plot(kind='bar', ax=ax, title="Analysis Metrics")
        st.pyplot(fig)
        
        # Export the results as CSV
        if st.button("Export Results"):
            df.to_csv("call_analysis_results.csv")
            st.success("Results saved as call_analysis_results.csv")

