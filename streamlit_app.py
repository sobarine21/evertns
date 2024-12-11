import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

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
        elif "help" in text or "resolve" in text or "assist" in text:
            return "Resolution"
    return "General"

# Function to generate a summary and rationale of the agent's performance
def generate_summary_and_rationale(conversations):
    agent_assistance = False
    rationale = []
    
    for speaker, text in conversations:
        if speaker == "Agent":
            # Check for resolution-related words
            if "help" in text or "resolve" in text or "assist" in text:
                agent_assistance = True
                rationale.append("Agent provided a solution to the customer's issue.")
                
            # Check if agent apologizes or shows empathy
            elif "apologize" in text or "understand" in text:
                agent_assistance = True
                rationale.append("Agent showed empathy by apologizing and acknowledging the customer's frustration.")
                
            # Check for compensation or assurance
            elif "compensation" in text or "credits" in text:
                agent_assistance = True
                rationale.append("Agent offered compensation or reassurance to the customer.")
                
    if agent_assistance:
        return "The agent was able to assist the customer.", rationale
    else:
        return "The agent was unable to truly assist the customer.", ["Agent failed to provide a solution or satisfactory resolution."]

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
        
        # Display the summary and rationale of whether the agent truly assisted or not
        summary, rationale = generate_summary_and_rationale(conversations)
        st.write("### Agent Assistance Summary")
        st.write(summary)
        
        # Display the rationale for the agent's assistance
        st.write("### Rationale for Assistance:")
        for reason in rationale:
            st.write(f"- {reason}")
        
        # Display charts for analysis
        st.write("Metrics Overview")
        fig, ax = plt.subplots()
        df.count().plot(kind='bar', ax=ax, title="Analysis Metrics")
        st.pyplot(fig)
        
        # Export the results as CSV
        if st.button("Export Results"):
            df.to_csv("call_analysis_results.csv")
            st.success("Results saved as call_analysis_results.csv")
