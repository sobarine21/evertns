import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

# Configure the API key securely
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize an empty DataFrame for storing results
results_df = pd.DataFrame(columns=["Agent Name", "Call Sentiment", "Resolution Status", "Empathy Score", "Clarity Score", "Resolution Score"])

# Streamlit App
st.title("Comprehensive Call Analysis Dashboard")
st.write("Upload call transcripts to analyze agent performance and generate support metrics.")

# File Upload
uploaded_files = st.file_uploader("Upload Call Transcript(s)", type=["txt"], accept_multiple_files=True)

# Function to analyze a single transcript
def analyze_transcript(transcript, agent_name="Unknown"):
    prompt = f"""
    Analyze this customer service call transcript. Provide the following:
    1. Overall sentiment (Positive, Neutral, Negative).
    2. Key issues discussed.
    3. Resolution status (Resolved/Unresolved).
    4. Support metrics ratings: Empathy (1-5), Clarity (1-5), Resolution (1-5).
    Call Transcript:
    {transcript}
    """
    try:
        # Load the Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Parse response (ensure this matches your AI's response format)
        lines = response.text.splitlines()
        sentiment = [line.split(":")[1].strip() for line in lines if "Sentiment" in line][0]
        resolution_status = [line.split(":")[1].strip() for line in lines if "Resolution Status" in line][0]
        empathy_score = int([line.split(":")[1].strip() for line in lines if "Empathy" in line][0])
        clarity_score = int([line.split(":")[1].strip() for line in lines if "Clarity" in line][0])
        resolution_score = int([line.split(":")[1].strip() for line in lines if "Resolution" in line][0])
        
        # Return metrics as a dictionary
        return {
            "Agent Name": agent_name,
            "Call Sentiment": sentiment,
            "Resolution Status": resolution_status,
            "Empathy Score": empathy_score,
            "Clarity Score": clarity_score,
            "Resolution Score": resolution_score
        }
    except Exception as e:
        return {"Error": str(e)}

if uploaded_files:
    st.subheader("Uploaded Transcripts")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
    
    # Analyze each uploaded transcript
    if st.button("Analyze All Transcripts"):
        with st.spinner("Analyzing transcripts..."):
            for uploaded_file in uploaded_files:
                transcript_text = uploaded_file.read().decode("utf-8")
                # Assume agent name is part of the file name (e.g., "AgentName_Call1.txt")
                agent_name = uploaded_file.name.split("_")[0] if "_" in uploaded_file.name else "Unknown"
                analysis_result = analyze_transcript(transcript_text, agent_name)
                if "Error" not in analysis_result:
                    results_df = pd.concat([results_df, pd.DataFrame([analysis_result])], ignore_index=True)

        st.success("Analysis completed!")

        # Display results
        st.subheader("Analysis Results")
        st.dataframe(results_df)

        # Visualizations
        st.subheader("Support Metrics Visualization")
        if not results_df.empty:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Average scores
            avg_empathy = results_df["Empathy Score"].mean()
            avg_clarity = results_df["Clarity Score"].mean()
            avg_resolution = results_df["Resolution Score"].mean()

            # Plot empathy
            ax[0].bar(["Empathy"], [avg_empathy], color="blue")
            ax[0].set_ylim(0, 5)
            ax[0].set_title("Average Empathy Score")

            # Plot clarity
            ax[1].bar(["Clarity"], [avg_clarity], color="green")
            ax[1].set_ylim(0, 5)
            ax[1].set_title("Average Clarity Score")

            # Plot resolution
            ax[2].bar(["Resolution"], [avg_resolution], color="orange")
            ax[2].set_ylim(0, 5)
            ax[2].set_title("Average Resolution Score")

            st.pyplot(fig)

        # Export results to Excel
        st.subheader("Export Results")
        if st.button("Download as Excel"):
            results_file = "call_analysis_results.xlsx"
            results_df.to_excel(results_file, index=False)
            st.success("Results exported! Download your file below.")
            st.download_button("Download Excel", data=open(results_file, "rb").read(), file_name=results_file)

# Footer
st.write("Powered by Google Gemini AI")
