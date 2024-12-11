import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Streamlit App UI - Beautiful header and custom theme
st.set_page_config(page_title="Customer Support Transcript Analyzer", layout="wide")
st.title("Customer Support Performance Analysis")
st.markdown("""
    Analyze agent-customer interactions, detect emotions, evaluate empathy and assistance, and identify improvement areas.
    **Just upload the call transcript below!**
""")
st.markdown("___")

# File Upload Section
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

# Function to analyze tone using TextBlob and more refined analysis
def analyze_tone(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Function to classify empathy and context
def classify_empathy_and_context(text):
    if any(word in text.lower() for word in ["sorry", "apologize", "understand", "feel", "sympathize"]):
        return "Empathy"
    elif any(word in text.lower() for word in ["help", "assist", "resolve", "solution"]):
        return "Assistance"
    else:
        return "Neutral"

# Function to analyze conversation context with advanced NLP
def analyze_conversation(text, speaker):
    if speaker == "Agent":
        if any(word in text.lower() for word in ["help", "assist", "resolve"]):
            return "Solution"
        elif any(word in text.lower() for word in ["sorry", "understand"]):
            return "Empathy"
    return "General"

# Function to analyze keywords and emotional patterns
def emotion_keywords(text):
    positive_keywords = ["thanks", "grateful", "appreciate", "pleased", "happy"]
    negative_keywords = ["frustrated", "angry", "disappointed", "sorry", "upset"]
    
    if any(word in text.lower() for word in positive_keywords):
        return "Positive Emotion"
    elif any(word in text.lower() for word in negative_keywords):
        return "Negative Emotion"
    return "Neutral Emotion"

# Function to generate a comprehensive analysis summary and rationale
def generate_summary_and_rationale(conversations):
    assistance_provided = False
    empathy_shown = False
    rationale = []
    
    for speaker, text in conversations:
        if speaker == "Agent":
            # Check for assistance or resolution-related keywords
            if "help" in text or "resolve" in text or "assist" in text:
                assistance_provided = True
                rationale.append(f"Agent provided assistance or resolution: '{text}'")
                
            # Check if agent apologized or showed empathy
            elif "sorry" in text or "understand" in text:
                empathy_shown = True
                rationale.append(f"Agent showed empathy: '{text}'")
                
            # Check for negative sentiment or failure to resolve
            elif "cannot" in text or "unable" in text:
                rationale.append(f"Agent was unable to resolve: '{text}'")
                
    if assistance_provided:
        return "The agent was able to assist the customer effectively.", rationale
    elif empathy_shown:
        return "The agent showed empathy but did not fully resolve the issue.", rationale
    else:
        return "The agent was unable to truly assist the customer.", ["Agent failed to provide a solution or resolution."]

# Display and process transcript if uploaded
if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode("utf-8")
    conversations = parse_transcript(transcript_text)
    
    # Show the conversations in an interactive way
    st.write("### Full Transcript")
    for speaker, text in conversations:
        st.write(f"**{speaker}:** {text}")
    
    # Button to analyze the transcript
    if st.button("Analyze Transcript"):
        # Initialize analysis metrics
        metrics = {
            "Empathy": [],
            "Assistance": [],
            "Tone": [],
            "Profanity": [],
            "Emotion": [],
            "Context": []
        }
        
        # Analyze each conversation
        for speaker, text in conversations:
            try:
                profanity = check_profanity(text)
                tone = analyze_tone(text)
                context = analyze_conversation(text, speaker)
                emotion = emotion_keywords(text)
                empathy_context = classify_empathy_and_context(text)
                
                # Update metrics
                metrics["Profanity"].append("Yes" if profanity else "No")
                metrics["Tone"].append(tone)
                metrics["Emotion"].append(emotion)
                metrics["Context"].append(context)
                metrics["Empathy"].append("Yes" if empathy_context == "Empathy" else "No")
                metrics["Assistance"].append("Yes" if empathy_context == "Assistance" else "No")
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Show metrics in a table
        df = pd.DataFrame(metrics)
        st.write(df)
        
        # Generate and display the summary and rationale
        summary, rationale = generate_summary_and_rationale(conversations)
        st.write("### Agent Assistance Summary")
        st.write(summary)
        
        # Show rationale with detailed context
        st.write("### Rationale for Assistance:")
        for reason in rationale:
            st.write(f"- {reason}")
        
        # Visual representation of metrics (using Seaborn for elegant visualizations)
        st.write("### Visual Analytics")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="Tone", palette="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Emotion Word Cloud
        st.write("### Emotion Word Cloud")
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(df["Emotion"]))
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        
        # Export the results as CSV
        if st.button("Export Results"):
            df.to_csv("call_analysis_results.csv", index=False)
            st.success("Results have been saved as 'call_analysis_results.csv'")
