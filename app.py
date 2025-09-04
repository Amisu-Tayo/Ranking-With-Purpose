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
    url = 'https://raw.githubusercontent.com/Amisu-Tayo/Ranking-With-Purpose/main/college_rankings_final_with_insights.csv'
    df = pd.read_csv(url, engine='python')
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

  # --- School Clusters ---
    st.markdown("---")
    st.header("Explore the Institutional Groups")
    st.markdown("""
    Based on their performance across all categories, the 700+ public colleges were grouped into four distinct clusters.
    Each group represents a unique operational profile. Click on a group to see the schools within it.
    """)

    # Get the unique, non-null cluster names and sort them
    cluster_names = sorted([name for name in df['cluster_name'].unique() if pd.notna(name)])

    for name in cluster_names:
        # Use an expander to neatly tuck away the list of schools for each cluster
        with st.expander(f"**{name}**"):
            st.markdown(f"### Schools in the '{name}' Group")
            cluster_df = df[df['cluster_name'] == name]
            
            # Select relevant columns to display
            display_cols = ['Institution Name', 'State', 'student_success_percentile', 'affordability_percentile', 'resources_percentile', 'equity_percentile', 'Insight']
            
            # Format percentiles for better readability (e.g., 0.87 -> 87.0%)
            # We also rename the columns for a cleaner look in the table header
            display_df = cluster_df[display_cols].rename(columns={
                'student_success_percentile': 'Success %',
                'affordability_percentile': 'Affordability %',
                'resources_percentile': 'Resources %',
                'equity_percentile': 'Equity %'
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

  

except FileNotFoundError:
    st.error("Error: The data file could not be found. Please ensure the URL in the `load_data` function is the correct 'Raw' link from your GitHub repository.")
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")

