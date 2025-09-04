import streamlit as st
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Ranking with Purpose",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the final data from GitHub using the robust Python engine."""
    url = 'https://raw.githubusercontent.com/Haleemah-Amisu/purposeful-rankings/main/college_rankings_final_with_insights.csv'
    # The engine='python' fix is essential for reading your specific CSV file.
    df = pd.read_csv(url, engine='python')
    return df

# --- Main App ---
try:
    df = load_data()

    # --- Header ---
    st.title('Ranking with Purpose: A New Lens on College Evaluation')
    st.markdown("""
    This tool moves beyond traditional metrics to highlight institutions that deliver strong outcomes, 
    provide access, and operate efficiently. Use the features below to find a school that truly fits your priorities.
    """)

    # --- School Recommender ---
    st.markdown("---")
    st.header("🎯 School Recommender")
    st.markdown("Use the sliders to set your minimum percentile for each category. The app will find schools that meet all your selected criteria.")

    ranking_metrics = {
        'student_success_percentile': 'Student Success',
        'affordability_percentile': 'Affordability',
        'resources_percentile': 'Academic Resources',
        'equity_percentile': 'Access & Equity'
    }

    cols = st.columns(len(ranking_metrics))
    user_priorities = {}

    for i, (metric, label) in enumerate(ranking_metrics.items()):
        with cols[i]:
            user_priorities[metric] = st.slider(
                f'Minimum {label} %', 0, 100, 50
            )

    filtered_df = df.copy()
    for metric, min_percentile in user_priorities.items():
        if metric in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[metric] >= min_percentile]

    st.subheader(f'{len(filtered_df)} Schools Match Your Criteria')
    
    # 'State' has been removed from this list of columns to display
    display_cols = ['Institution Name'] + list(ranking_metrics.keys())
    st.dataframe(
        filtered_df[display_cols].rename(columns={k: f"{v} %" for k, v in ranking_metrics.items()}),
        use_container_width=True,
        hide_index=True
    )

    # --- Institutional Groups Explorer ---
    st.markdown("---")
    st.header("Explore the Institutional Groups")
    st.markdown("Colleges were grouped into four distinct clusters based on their overall performance profiles.")

    if 'cluster_name' in df.columns:
        cluster_names = sorted([name for name in df['cluster_name'].unique() if pd.notna(name)])

        for name in cluster_names:
            with st.expander(f"**{name}**"):
                st.markdown(f"### Schools in the '{name}' Group")
                cluster_df = df[df['cluster_name'] == name]
                
                # 'State' has been removed from this list of columns to display
                display_cols = ['Institution Name', 'Insight'] + list(ranking_metrics.keys())
                
                st.dataframe(
                    cluster_df[display_cols].rename(columns={k: f"{v} %" for k, v in ranking_metrics.items()}),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.warning("The 'cluster_name' column was not found in the data.")

except Exception as e:
    st.error(f"An unexpected error occurred. Error details: {e}")

