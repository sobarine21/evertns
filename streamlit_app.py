import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Customer Support Transcript Analyzer")
st.write("Analyze call transcripts and get actionable insights into the quality of the conversation.")

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
        metrics = {"Empathy": [], "Clarity": [], "Resolution": []}
        
        # Loop through each conversation pair and analyze with Gemini AI
        for speaker, text in conversations:
            try:
                # Generate analysis for each message
                prompt = f"Analyze the following text for empathy, clarity, and resolution: {text}."
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                # Mock-up for processing AI's response
                empathy = "High" if "understand" in response.text else "Low"
                clarity = "High" if "clear" in response.text else "Low"
                resolution = "Resolved" if "resolved" in response.text else "Unresolved"
                
                metrics["Empathy"].append(empathy)
                metrics["Clarity"].append(clarity)
                metrics["Resolution"].append(resolution)
            
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

