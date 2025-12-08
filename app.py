import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

# --- API Key from Streamlit Secrets ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# --- import Recommender model ---
from rwp_recommender import (
    FEATURE_COLS,
    recommend_cluster_aware,
    normalize_weights,
)

# --- Helper: clean percentile values ---
def safe_percentile(val):
    """Return rounded percentile (0–100) or None if invalid."""
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None


# --- LLM call for final HawkSight advice ---
def call_hawksight_llm(prompt: str) -> str:
    """
    Calls Groq's Llama 3 model to power HawkSight.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY is not set in Streamlit secrets."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are HawkSight, a warm, practical college advisor. "
                    "You help high school students and their families think about good-fit public colleges "
                    "based on data that is shown to you. "
                    "Use simple, encouraging language. Avoid jargon. Don't explain how the data was built; "
                    "just use it to give grounded, honest advice."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"HawkSight ran into an error while generating advice: {e}"


# --- LLM call to interpret the student's description into preferences ---
def parse_student_profile_with_llm(description: str):
    """
    Use Groq (Llama 3) to convert a natural-language student description
    into structured preferences for the recommender system.

    Returns (weights, thresholds, states_preferred).
    """
    if not GROQ_API_KEY:
        # Fallback: equal weights, no filters
        default_weights = {
            "student_success_percentile": 0.25,
            "affordability_percentile": 0.25,
            "resources_percentile": 0.25,
            "equity_percentile": 0.25,
        }
        return normalize_weights(default_weights), {}, []

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = """
You are helping configure a college recommender system.

Your job is to read a student's description and output a JSON object with:
{
  "weights": {
    "student_success_percentile": float,
    "affordability_percentile": float,
    "resources_percentile": float,
    "equity_percentile": float
  },
  "thresholds": {
    "...": float or omit if not needed
  },
  "states_preferred": ["two-letter codes like 'MD', 'NY'"] 
}

Guidelines:
- Weights reflect what the student *cares about most* (higher = more important).
- You do NOT need weights to sum to 1; they will be normalized later.
- Use thresholds only when the student clearly wants to avoid low values 
  (e.g. very low affordability, very low success).
- Include states_preferred if the student mentions specific states or locations.
Return ONLY valid JSON. No comments or extra text.
"""

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Student description:\n{description}",
            },
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # In case it wraps JSON in ```json ``` fences
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        parsed = json.loads(content)

        weights = parsed.get("weights", {})
        thresholds = parsed.get("thresholds", {}) or {}
        states_preferred = parsed.get("states_preferred", []) or []

        weights = normalize_weights(weights)

        # Ensure states_preferred is a list of strings
        states_preferred = [str(s).upper() for s in states_preferred]

        return weights, thresholds, states_preferred

    except Exception as e:
        # Fallback if parsing / API fails
        default_weights = {
            "student_success_percentile": 0.25,
            "affordability_percentile": 0.25,
            "resources_percentile": 0.25,
            "equity_percentile": 0.25,
        }
        print("parse_student_profile_with_llm error:", e)
        return normalize_weights(default_weights), {}, []


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
    url = 'https://raw.githubusercontent.com/Amisu-Tayo/Ranking-With-Purpose/refs/heads/main/college_rankings_with_efficiency.csv'
    df = pd.read_csv(url, engine='python')
    return df


# --- Main App ---
try:
    df = load_data()

    # --- Header ---
    st.title('🎯 Ranking with Purpose')
    st.markdown("A new lens on college evaluation, designed to help you find a school that's the right fit for *you*.")

    # --- Create Tabs for a Clean Interface ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Build Your Ranking",
        "🔭 Explore Groups",
        "🔍 Find a School",
        "🗺️ Cluster Map",
        "🦅 Talk to HawkSight"
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
            filtered_df[['Institution Name', 'State'] + list(ranking_metrics.keys())]
                .rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
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
                        cluster_df[['Institution Name', 'State', 'Insight'] + list(ranking_metrics.keys())]
                            .rename(columns={k: f"{v} (Percentile)" for k, v in ranking_metrics.items()}),
                        use_container_width=True, hide_index=True
                    )

    # --- Tab 3: Find a School ---
    with tab3:
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
                st.markdown("**Core Rankings**", help="How this school ranks against all others. A rank of 90 means it's in the top 10%.")
                for metric, label in ranking_metrics.items():
                    if metric in school and pd.notna(school[metric]):
                        st.metric(label=f"{label} (Percentile Rank)", value=f"{school[metric]:.1f}")

            with res_col2:
                st.markdown("**Efficiency Metrics**", help="A score from 0-100 showing how this school's efficiency compares to others. A higher score is better.")

                efficiency_metrics_map = {
                    'grad_per_instruction_percentile': 'Graduation Rate per Dollar Spent on Instruction',
                    'retention_per_services_percentile': 'Retention Rate per Dollar Spent on Student Services',
                    'degrees_per_netprice_percentile': 'Graduation Rate per Dollar of Net Price',
                    'grad_per_core_exp_percentile': 'Graduation Rate per Dollar of Core Expenses',
                    'degrees_per_endowment_percentile': 'Graduation Rate per Dollar of Endowment',
                    'grad_per_faculty_percentile': 'Graduation Rate per Full-Time Faculty Member',
                    'retention_per_revenue_percentile': 'Retention Rate per Dollar of Total Revenue'
                }

                for metric_col, friendly_name in efficiency_metrics_map.items():
                    if metric_col in school and pd.notna(school[metric_col]):
                        st.metric(label=f"{friendly_name} (Efficiency Score)", value=f"{school[metric_col]:.1f}")

            with res_col3:
                st.markdown("**Key Individual Stats**", help="A few important raw data points for this school.")

                def pct_str(v):
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        return None
                    if x <= 1.0:
                        x *= 100
                    return f"{x:.1f}%"

                if 'Graduation Rate (4yr)' in school and pd.notna(school['Graduation Rate (4yr)']):
                    st.metric(label="4-Year Graduation Rate", value=f"{school['Graduation Rate (4yr)']:.1f}%")

                val = school.get("Graduation Rate (5yr)")
                if pd.isna(val):
                    val = school.get("Graduation rate - Bachelor degree within 5 years  total (DRVGR2023)")
                if pd.notna(val):
                    st.metric(label="5-Year Graduation Rate", value=pct_str(val))

                val = school.get("Graduation Rate (6yr)")
                if pd.isna(val):
                    val = school.get("Graduation rate - Bachelor degree within 6 years  total (DRVGR2023)")
                if pd.notna(val):
                    st.metric(label="6-Year Graduation Rate", value=pct_str(val))

                if 'Retention Rate' in school and pd.notna(school['Retention Rate']):
                    st.metric(label="Full-Time Retention Rate", value=f"{school['Retention Rate']:.1f}%")

                if 'Student-to-Faculty Ratio' in school and pd.notna(school['Student-to-Faculty Ratio']):
                    st.metric(label="Student-to-Faculty Ratio", value=f"{int(school['Student-to-Faculty Ratio'])} to 1")

                if 'Average Net Price' in school and pd.notna(school['Average Net Price']):
                    st.metric(label="Average Net Price", value=f"${int(school['Average Net Price']):,}")

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
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
        st.caption("This chart uses a technique called PCA to represent the four complex ranking dimensions on a simple 2D map, revealing the hidden structure in the data.")

    # --- Tab 5: HawkSight Advisor ---
    with tab5:
        st.header("🦅 HawkSight — Precision Guidance for College Decisions")
        st.markdown(
            "**HawkSight** evaluates universities the way a hawk surveys terrain — with clarity, focus, "
            "and an instinct for the strongest landing point 🎯.\n\n"
            "Describe a student, and HawkSight will identify compatible institutions using "
            "your RWP metrics (Success, Affordability, Resources, Equity) under the hood."
        )

        left_col, right_col = st.columns([1, 1.2])

        with left_col:
            student_description = st.text_area(
                "Who is this student and what do they care about?",
                placeholder=(
                    "Example: I'm a first-generation student interested in biology. "
                    "My family can't afford high tuition, and I care a lot about graduation rates "
                    "and feeling supported on campus. I'd like to stay in MD or NY."
                ),
                height=140,
            )

            num_recs = st.number_input(
                "How many colleges should HawkSight recommend?",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )

            run_button = st.button("Get guidance from HawkSight")

        with right_col:
            st.subheader("HawkSight Recommendation")

            if run_button:
                if not student_description.strip():
                    st.warning("Please describe the student first.")
                else:
                    # 1) Let the LLM parse preferences → weights / thresholds / states
                    weights, thresholds, states_pref = parse_student_profile_with_llm(
                        student_description
                    )

                    with st.expander("See how HawkSight interpreted this profile"):
                        st.write("**Weights (normalized):**", weights)
                        st.write("**Thresholds:**", thresholds)
                        st.write("**Preferred states:**", states_pref)

                    # 2) Use the cluster-aware recommender to pick colleges (internal pool = 30)
                    recommended = recommend_cluster_aware(
                        df,
                        base_model="weighted",
                        weights=weights,
                        thresholds=thresholds,
                        states_preferred=states_pref,
                        states_excluded=None,
                        top_k=30,
                    )

                    # Only keep as many as the user asked to see
                    recommended = recommended.head(int(num_recs))

                    # 3) Build a compact text context for the LLM
                    lines = []
                    for _, row in recommended.iterrows():
                        try:
                            s = safe_percentile(row.get("student_success_percentile"))
                            a = safe_percentile(row.get("affordability_percentile"))
                            r = safe_percentile(row.get("resources_percentile"))
                            e = safe_percentile(row.get("equity_percentile"))
                        except Exception:
                            continue

                        if None in (s, a, r, e):
                            continue

                        lines.append(
                            f"{row['Institution Name']} ({row['State']}) — "
                            f"Success: {s}th percentile, "
                            f"Affordability: {a}th percentile, "
                            f"Resources: {r}th percentile, "
                            f"Equity: {e}th percentile"
                        )

                    context_table = "\n".join(lines)

                    prompt = f"""
The list below contains public colleges that were ALREADY selected
by a recommender system based on the student's description.

Each line shows one college, its state, and percentile rankings (0–100):

{context_table}

Now, here is the student:

\"\"\"{student_description}\"\"\"

Using only the colleges listed above:

- Suggest up to {int(num_recs)} colleges that could be a good fit for this student.
- Explain why in simple language, focusing on what matters to them 
  (money, outcomes, support, location, etc.).
- Mention trade-offs honestly (for example, one might be cheaper while another has stronger outcomes).
- Do not invent new data that isn't in the list.
- Speak directly to the student or their family, and be warm but honest.
"""

                    with st.spinner("HawkSight is scanning the field..."):
                        advice = call_hawksight_llm(prompt)

                    st.write(advice)

                    # 4) Visual profile for the top match (no overall score)
                    if not recommended.empty:
                        top_school = recommended.iloc[0]

                        s = safe_percentile(top_school.get("student_success_percentile"))
                        a = safe_percentile(top_school.get("affordability_percentile"))
                        r = safe_percentile(top_school.get("resources_percentile"))
                        e = safe_percentile(top_school.get("equity_percentile"))

                        metrics_for_chart = {}
                        if s is not None:
                            metrics_for_chart["Student Success"] = s
                        if a is not None:
                            metrics_for_chart["Affordability"] = a
                        if r is not None:
                            metrics_for_chart["Resources"] = r
                        if e is not None:
                            metrics_for_chart["Equity"] = e

                        if metrics_for_chart:
                            st.subheader(
                                f"RWP Profile for Top Match: {top_school['Institution Name']} ({top_school['State']})"
                            )

                            fig2, ax2 = plt.subplots()
                            ax2.bar(list(metrics_for_chart.keys()), list(metrics_for_chart.values()))
                            ax2.set_ylim(0, 100)
                            ax2.set_ylabel("Percentile Rank (0–100)")
                            ax2.set_title("How this school compares across RWP dimensions")
                            ax2.grid(axis="y", linestyle="--", alpha=0.4)

                            st.pyplot(fig2)
            else:
                st.caption("HawkSight's guidance will appear here after you click the button.")

except Exception as e:
    st.error(f"An unexpected error occurred. Error details: {e}")

