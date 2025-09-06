import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Page Configuration ---
st.set_page_config(
    page_title="Ranking with Purpose",
    layout="wide",
    page_icon="🎯"
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
    st.title('🎯 Ranking with Purpose')
    st.markdown("A new lens on college evaluation, designed to help you find a school that's the right fit for *you*.")

    # --- Create Four Tabs for a Clean Interface ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Build Your Ranking",
        "🔭 Explore Groups",
        "🔍 Find a School",
        "🗺️ Cluster Map"
    ])

    # --- Tab 1: Build Your Ranking ---
    with tab1:
        st.header("Find Schools That Match Your Priorities")
        st.info(
            """
            **How to use this tool:** Adjust the sliders below to set your priorities.
            A **percentile rank** shows how a school compares to others. Setting a slider to **80** means you're looking for schools
            that perform better than **80%** of all other public colleges in that category.
            """
        )

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
                    label, 0, 100, 50, key=f'slider_{metric}'
                )

        filtered_df = df.copy()
        for metric, min_percentile in user_priorities.items():
            if metric in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[metric] >= min_percentile]

        st.subheader(f'{len(filtered_df)} Schools Match Your Criteria')
        st.dataframe(
            filtered_df[['Institution Name'] + list(ranking_metrics.keys())].rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
            use_container_width=True, hide_index=True
        )

    # --- Tab 2: Explore Groups ---
    with tab2:
        st.header("Discover Different Types of High-Performing Institutions")
        st.markdown("Based on their overall profiles, colleges were sorted into four distinct groups. Click on a group to see the schools inside.")

        if 'cluster_name' in df.columns:
            cluster_names = sorted([name for name in df['cluster_name'].unique() if pd.notna(name)])
            for name in cluster_names:
                with st.expander(f"**{name}**"):
                    cluster_df = df[df['cluster_name'] == name]
                    st.dataframe(
                        cluster_df[['Institution Name', 'Insight'] + list(ranking_metrics.keys())].rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
                        use_container_width=True, hide_index=True
                    )

    # --- Tab 3: Find a School (FINAL CORRECTED VERSION) ---
    with tab3:
        st.header("Look Up a Specific School")
        search_term = st.text_input("Start typing a school name to search:")

        cleaned_search_term = search_term.strip()

        if cleaned_search_term:
            possible_matches_df = df[df['Institution Name'].str.contains(cleaned_search_term, case=False, na=False)]
            
            if not possible_matches_df.empty:
                possible_names = possible_matches_df['Institution Name'].tolist()
                options = ["-- Select a school from the list --"] + sorted(possible_names)
                selected_school_name = st.selectbox("Select a matching school:", options)

                if selected_school_name != "-- Select a school from the list --":
                    school = possible_matches_df[possible_matches_df['Institution Name'] == selected_school_name].iloc[0]
                    
                    st.markdown(f"### {school['Institution Name']}")
                    st.write(f"**Institutional Group:** {school['cluster_name']}")
                    st.info(f"**Insight:** *{school['Insight']}*")
                    st.markdown("---")
                    
                    st.subheader("Performance Snapshot")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.markdown("**Core Rankings**", help="How this school ranks against all others. A percentile rank of 90 means it's in the top 10%.")
                        for metric, label in ranking_metrics.items():
                            st.metric(label=f"{label} (Percentile Rank)", value=f"{school[metric]:.1f}")
                    
                    with res_col2:
                        st.markdown("**Efficiency Metrics**", help="A score from 0-100 showing how this school's efficiency compares to others. A higher score is better.")
                        
                        # --- BUG FIX #1: Corrected dictionary keys to match the final CSV ---
                        efficiency_metrics_map = {
                            'Graduation per Instructional Spending_percentile': 'Grads per Instruction $',
                            'Retention per Student Services Spending_percentile': 'Retention per Student Services $',
                            'Degrees per Net Price_percentile': 'Grads per Net Price $',
                            'Graduation per Core Expenses_percentile': 'Grads per Core Expenses $',
                            'Degrees per Endowment per FTE_percentile': 'Grads per Endowment $'
                        }
                        for metric_col, friendly_name in efficiency_metrics_map.items():
                             if metric_col in school and pd.notna(school[metric_col]):
                                st.metric(label=f"{friendly_name} (Percentile Rank)", value=f"{school[metric_col]:.1f}")

                    with res_col3:
                        st.markdown("**Key Individual Stats**", help="A few important raw data points for this school.")
                        
                        # --- BUG FIX #2: Code now correctly displays the true raw stats from the CSV ---
                        if 'Graduation Rate (4yr)' in school and pd.notna(school['Graduation Rate (4yr)']):
                             st.metric(label="4-Year Graduation Rate", value=f"{school['Graduation Rate (4yr)']:.1f}%")
                        if 'Graduation Rate (5yr)' in school and pd.notna(school['Graduation Rate (5yr)']):
                             st.metric(label="5-Year Graduation Rate", value=f"{school['Graduation Rate (5yr)']:.1f}%")
                        if 'Retention Rate' in school and pd.notna(school['Retention Rate']):
                            st.metric(label="Full-Time Retention Rate", value=f"{school['Retention Rate']:.1f}%")
                        if 'Student-to-Faculty Ratio' in school and pd.notna(school['Student-to-Faculty Ratio']):
                            st.metric(label="Student-to-Faculty Ratio", value=f"{int(school['Student-to-Faculty Ratio'])} to 1")
                        if 'Average Net Price' in school and pd.notna(school['Average Net Price']):
                            st.metric(label="Average Net Price", value=f"${int(school['Average Net Price']):,}")

            else:
                st.warning("No schools found matching that name.")

    # --- Tab 4: Cluster Map ---
    with tab4:
        st.header("Visualize the College Landscape")
        st.markdown("This map shows all institutions plotted based on their overall profile. Each color represents one of the four institutional groups.")

        pca_features = ['student_success_score', 'affordability_score', 'resources_score', 'equity_score']
        pca_df = df.dropna(subset=pca_features + ['cluster_name'])
        
        X = StandardScaler().fit_transform(pca_df[pca_features])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        pca_df['pca1'] = X_pca[:, 0]
        pca_df['pca2'] = X_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        clusters = sorted(pca_df['cluster_name'].unique())
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
        st.caption("This chart uses a technique called PCA to represent the four complex ranking dimensions on a simple 2D map, revealing the hidden structure in the data.")

except Exception as e:
    st.error(f"An unexpected error occurred. Please ensure your CSV file is up to date and accessible at the specified URL. Error details: {e}")

