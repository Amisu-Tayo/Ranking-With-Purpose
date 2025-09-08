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
    """Loads the final, complete data from GitHub and renames key columns for easier use."""
    url = 'https://raw.githubusercontent.com/Amisu-Tayo/Ranking-With-Purpose/main/college_rankings_final_with_insights.csv'
    df = pd.read_csv(url, engine='python')
    # Rename state column to ensure consistency across the app
    if 'State abbreviation (HD2023)' in df.columns:
        df.rename(columns={'State abbreviation (HD2023)': 'State'}, inplace=True)
    return df

# --- Main App ---
try:
    df = load_data()

    # --- Header ---
    st.title('🎯 Ranking with Purpose')
    st.markdown("A new lens on college evaluation, designed to help you find a school that's the right fit for *you*.")

    # --- Navigation Bar ---
    selected_tab = st.radio(
        "Navigation",
        ["📊 Build Your Ranking", "🔭 Explore Groups", "🔍 Find a School", "🗺️ Cluster Map"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # --- Tab 1: Build Your Ranking ---
    if selected_tab == "📊 Build Your Ranking":
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
            filtered_df[['Institution Name', 'State'] + list(ranking_metrics.keys())].rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
            use_container_width=True, hide_index=True
        )

    # --- Tab 2: Explore Groups ---
    if selected_tab == "🔭 Explore Groups":
        st.header("Discover Different Types of High-Performing Institutions")
        st.markdown("Based on their overall profiles, colleges were sorted into four distinct groups. Click on a group to see the schools inside.")
        if 'cluster_name' in df.columns:
            cluster_names = sorted([name for name in df['cluster_name'].unique() if pd.notna(name)])
            for name in cluster_names:
                with st.expander(f"**{name}**"):
                    cluster_df = df[df['cluster_name'] == name]
                    st.dataframe(
                        cluster_df[['Institution Name', 'State', 'Insight'] + list(ranking_metrics.keys())].rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
                        use_container_width=True, hide_index=True
                    )

    # --- Tab 3: Find a School ---
    if selected_tab == "🔍 Find a School":
        st.header("Look Up a Specific School")
        school_list = ["-- Select a school --"] + sorted(df['Institution Name'].unique())
        selected_school_name = st.selectbox("Search for a school by typing its name below:", school_list)

        if selected_school_name != "-- Select a school --":
            school = df[df['Institution Name'] == selected_school_name].iloc[0]
            st.markdown(f"### {school['Institution Name']}, {school['State']}")
            st.write(f"**Institutional Group:** {school['cluster_name']}")
            st.info(f"**Insight:** *{school['Insight']}*")
            st.markdown("---")
            st.subheader("Performance Snapshot")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                ranking_metrics = {
                    'student_success_percentile': 'Student Success',
                    'affordability_percentile': 'Affordability',
                    'resources_percentile': 'Academic Resources',
                    'equity_percentile': 'Access & Equity'
                }
                st.markdown("**Core Rankings**", help="How this school ranks against all others.")
                for metric, label in ranking_metrics.items():
                    st.metric(label=f"{label} (Percentile Rank)", value=f"{school[metric]:.1f}")
            with res_col2:
                st.markdown("**Efficiency Metrics**", help="A score from 0-100 showing efficiency compared to others.")
                
                # Added the efficiency metrics with corrected column names
                efficiency_metrics_map = {
                    'Graduation Rate per Instructional Spending_percentile': 'Grads per Instruction $',
                    'Retention Rate per Student Service Spending_percentile': 'Retention per Student Services $',
                    'Graduation Rate per Net Price_percentile': 'Grads per Net Price $',
                    'Graduation Rate per Core Expenses_percentile': 'Grads per Core Expenses $',
                    'Graduation Rate per Endowment per FTE_percentile': 'Grads per Endowment $'
                }
                for metric_col, friendly_name in efficiency_metrics_map.items():
                        if metric_col in school and pd.notna(school[metric_col]):
                            st.metric(label=f"{friendly_name} (Efficiency Score)", value=f"{school[metric_col]:.1f}")
            with res_col3:
                st.markdown("**Key Individual Stats**", help="A few important raw data points for this school.")
                
                if '4-year Grad Rate' in school and pd.notna(school['4-year Grad Rate']):
                        st.metric(label="4-Year Graduation Rate", value=f"{school['4-year Grad Rate']:.1f}%")
                # Added 5-year and 6-year grad rates
                if '5-year Grad Rate' in school and pd.notna(school['5-year Grad Rate']):
                        st.metric(label="5-Year Graduation Rate", value=f"{school['5-year Grad Rate']:.1f}%")
                if '6-year Grad Rate' in school and pd.notna(school['6-year Grad Rate']):
                        st.metric(label="6-Year Graduation Rate", value=f"{school['6-year Grad Rate']:.1f}%")
                if 'Student to Faculty Ratio' in school and pd.notna(school['Student to Faculty Ratio']):
                    st.metric(label="Student-to-Faculty Ratio", value=f"{int(school['Student to Faculty Ratio'])} to 1")

    # --- Tab 4: Cluster Map ---
    if selected_tab == "🗺️ Cluster Map":
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
        colors = plt.get_cmap('viridis')(range(len(clusters)))
        for i, cluster in enumerate(clusters):
            cluster_data = pca_df[pca_df['cluster_name'] == cluster]
            ax.scatter(cluster_data['pca1'], cluster_data['pca2'], color=colors[i], label=cluster, alpha=0.7)
        ax.set_title('Institutional Cluster Map')
        ax.set_xlabel('Principal Component 1 (Success vs. Affordability/Equity)')
        ax.set_ylabel('Principal Component 2 (Academic Resources)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.caption("This chart uses a technique called PCA to represent the four complex ranking dimensions on a simple 2D map.")

except Exception as e:
    st.error(f"An unexpected error occurred. Please ensure your CSV file is up to date and accessible at the specified URL. Error details: {e}")

