import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Page Configuration ---
st.set_page_config(
    page_title="Ranking with Purpose",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the final, complete data from GitHub using the robust Python engine."""
    url = 'https://raw.githubusercontent.com/Amisu-Tayo/Ranking-With-Purpose/main/college_rankings_final_with_insights.csv'
    df = pd.read_csv(url, engine='python')
    return df

# --- Main App ---
try:
    df = load_data()

    # --- Header ---
    st.title('Ranking with Purpose: A New Lens on College Evaluation')
    st.markdown("This tool will find you a school that is perfectly tailored to YOUR needs.")

    # --- Create Four Tabs for a Clean Interface ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 School Recommender",
        "🔭 Explore Groups",
        "🔍 Find a School",
        "🗺️ Cluster Map"
    ])

    # --- Tab 1: School Recommender ---
    with tab1:
        st.header("Find a School That Fits Your Priorities")
        st.markdown("Use the sliders to set your minimum percentile for each category. The table will update to show schools that meet all your criteria.")

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
                    f'Minimum {label} %', 0, 100, 50, key=f'slider_{metric}'
                )

        filtered_df = df.copy()
        for metric, min_percentile in user_priorities.items():
            if metric in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[metric] >= min_percentile]

        st.subheader(f'{len(filtered_df)} Schools Match Your Criteria')
        display_cols = ['Institution Name'] + list(ranking_metrics.keys())
        st.dataframe(
            filtered_df[display_cols].rename(columns={k: f"{v} %" for k, v in ranking_metrics.items()}),
            use_container_width=True, hide_index=True
        )

    # --- Tab 2: Explore Institutional Groups ---
    with tab2:
        st.header("Discover Different Types of High-Performing Institutions")
        st.markdown("Colleges were grouped into four distinct clusters based on their overall performance profiles.")

        if 'cluster_name' in df.columns:
            cluster_names = sorted([name for name in df['cluster_name'].unique() if pd.notna(name)])
            for name in cluster_names:
                with st.expander(f"**{name}**"):
                    cluster_df = df[df['cluster_name'] == name]
                    st.dataframe(
                        cluster_df[['Institution Name', 'Insight'] + list(ranking_metrics.keys())].rename(columns={k: f"{v} %" for k, v in ranking_metrics.items()}),
                        use_container_width=True, hide_index=True
                    )

    # --- Tab 3: Find a School (with NEW, Improved Search UX) ---
    with tab3:
        st.header("Look Up a Specific School")
        search_term = st.text_input("Start typing a school name:")

        # --- NEW 2-STEP SEARCH LOGIC ---
        cleaned_search_term = search_term.strip()

        if cleaned_search_term:
            # Step 1: Find all possible matches and populate a dropdown
            possible_matches = df[df['Institution Name'].str.contains(cleaned_search_term, case=False, na=False)]['Institution Name'].tolist()
            
            if possible_matches:
                # Add a placeholder to the beginning of the list
                options = ["-- Select a school --"] + sorted(possible_matches)
                selected_school_name = st.selectbox("Did you mean...?", options)

                # Step 2: If the user selects a valid school, show its profile
                if selected_school_name != "-- Select a school --":
                    school = df[df['Institution Name'] == selected_school_name].iloc[0]
                    
                    st.markdown(f"#### {school['Institution Name']}")
                    st.write(f"**Insight:** {school['Insight']}")
                    st.write(f"**Institutional Group:** {school['cluster_name']}")
                    st.markdown("---")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.markdown("**Core Rankings (Percentile)**")
                        for metric, label in ranking_metrics.items():
                            st.write(f"{label}: **{school[metric]:.1f}%**")
                    with res_col2:
                        st.markdown("**Key Individual Stats**")
                        key_stats = ['Graduation Rate (4yr)', 'Retention Rate', 'Average Net Price', 'Student-to-Faculty Ratio', 'Pell Grant Percentage']
                        for stat in key_stats:
                            if stat in school and pd.notna(school[stat]):
                                st.write(f"{stat}: **{school[stat]:,.1f}**")
                    with res_col3:
                        st.markdown("**Efficiency Metrics (Percentile)**")
                        # This list comprehension correctly finds the efficiency columns
                        efficiency_metrics = [col for col in df.columns if 'per' in col and '_percentile' in col]
                        # --- BUG FIX: This loop now correctly iterates over the `efficiency_metrics` list ---
                        for metric in efficiency_metrics:
                            st.write(f"{metric.replace('_percentile', '').replace('_', ' ').title()}: **{school[metric]:.1f}%**")
            else:
                st.warning("No schools found matching that name.")

    # --- Tab 4: Cluster Map ---
    with tab4:
        st.header("Visualize the College Landscape")
        st.markdown("This map shows all 700+ institutions plotted based on their overall profile. Each color represents one of the four institutional groups, showing the distinct patterns in public higher education.")

        pca_features = ['student_success_score', 'affordability_score', 'resources_score', 'equity_score']
        pca_df = df.dropna(subset=pca_features + ['cluster_name'])
        
        X = StandardScaler().fit_transform(pca_df[pca_features])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        pca_df['pca1'] = X_pca[:, 0]
        pca_df['pca2'] = X_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        clusters = sorted(pca_df['cluster_name'].unique()) # Sort clusters for consistent color mapping
        colors = plt.cm.get_cmap('viridis', len(clusters))
        
        for i, cluster in enumerate(clusters):
            cluster_data = pca_df[pca_df['cluster_name'] == cluster]
            ax.scatter(cluster_data['pca1'], cluster_data['pca2'], color=colors(i), label=cluster, alpha=0.7)
            
        ax.set_title('Institutional Cluster Map')
        ax.set_xlabel('Principal Component 1 (Success vs. Affordability/Equity)')
        ax.set_ylabel('Principal Component 2 (Academic Resources)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"An unexpected error occurred. Please ensure your CSV file is up to date and accessible at the specified URL. Error details: {e}")

