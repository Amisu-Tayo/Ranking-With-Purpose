import streamlit as st
import pandas as pd

# --- Page Configuration ---
# Set the page title and a wide layout for the app
st.set_page_config(
    page_title="Ranking with Purpose",
    layout="wide"
)

# --- Data Loading ---
# Use st.cache_data to load the data once and store it in cache for better performance
@st.cache_data
def load_data():
    # Load the final CSV file from GitHub repository
    url = 'https://github.com/Amisu-Tayo/Ranking-With-Purpose/blob/main/college_rankings_final_with_insights.csv'
    df = pd.read_csv(url)
    return df

try:
    df = load_data()
    
    # --- Header ---
    st.title('Ranking with Purpose: A New Lens on College Evaluation')
    st.markdown("""
    Welcome to a new way of evaluating public colleges. This tool moves beyond traditional metrics
    to highlight institutions that deliver strong outcomes, provide access, and operate efficiently.
    Explore the data to find a school that truly fits your priorities.
    """)

  

except FileNotFoundError:
    st.error("Error: The data file could not be found. Please ensure the URL in the `load_data` function is the correct 'Raw' link from your GitHub repository.")
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")

