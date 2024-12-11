import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

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

# Function to analyze tone
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
        metrics = {"Empathy": [], "Clarity": [], "Resolution": [], "Tone": [], "Profanity": [], "Topic": []}
        
        # Loop through each conversation pair and analyze with Gemini AI
        for speaker, text in conversations:
            try:
                # Check for profanity
                profanity = check_profanity(text)
                tone = analyze_tone(text)
                topic = analyze_conversation(text, speaker)
                
                # Generate context-aware analysis with Google Gemini
                prompt = f"Analyze the following conversation for professionalism, empathy, tone, and context: {text}"
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                # Simulate AI response (can be improved by real analysis)
                empathy = "High" if "understand" in response.text else "Low"
                clarity = "High" if "clear" in response.text else "Low"
                resolution = "Resolved" if "resolved" in response.text else "Unresolved"
                
                metrics["Empathy"].append(empathy)
                metrics["Clarity"].append(clarity)
                metrics["Resolution"].append(resolution)
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

