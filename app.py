import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

# --- API Key from Streamlit Secrets ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)


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


def extract_preferences_with_llm(student_text: str) -> dict:
    """
    Uses Groq LLM to turn a free-text student description into
    numeric weights + optional thresholds and state preferences.

    Returns a Python dict. Falls back to equal weights on any failure.
    """
    if not GROQ_API_KEY:
        # fallback: equal weights
        return {
            "weights": {
                "student_success_percentile": 0.25,
                "affordability_percentile": 0.25,
                "resources_percentile": 0.25,
                "equity_percentile": 0.25,
            },
            "min_thresholds": {},
            "states_preferred": [],
            "states_excluded": [],
        }

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are HawkSight-Config, a helper that ONLY outputs JSON.\n"
        "Given a student's description, you will map it to preferences over four metrics:\n"
        "- student_success_percentile\n"
        "- affordability_percentile\n"
        "- resources_percentile\n"
        "- equity_percentile\n\n"
        "Output a JSON object with:\n"
        "{\n"
        "  \"weights\": { metric_name: float 0-1, ... },\n"
        "  \"min_thresholds\": { metric_name: float 0-100, ... },\n"
        "  \"states_preferred\": [\"MD\", \"PA\"],\n"
        "  \"states_excluded\": [\"CA\"]\n"
        "}\n"
        "Weights should roughly sum to 1. If the student doesn't care about location,\n"
        "use empty arrays for states. Do NOT include any extra keys. Do NOT write prose."
    )

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": student_text},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]

        # Try to parse JSON from the model output
        prefs = json.loads(raw_content)
        return prefs
    except Exception:
        # Any error: fall back to equal weights, no thresholds
        return {
            "weights": {
                "student_success_percentile": 0.25,
                "affordability_percentile": 0.25,
                "resources_percentile": 0.25,
                "equity_percentile": 0.25,
            },
            "min_thresholds": {},
            "states_preferred": [],
            "states_excluded": [],
        }


def score_schools_with_preferences(df_in: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    """
    Apply weights + thresholds + state filters to ALL schools,
    and compute a HawkSight score for each.
    Returns a new dataframe sorted by descending score.
    """
    metric_cols = [
        "student_success_percentile",
        "affordability_percentile",
        "resources_percentile",
        "equity_percentile",
    ]

    df_scored = df_in.copy()

    # 1. State filters
    states_pref = prefs.get("states_preferred") or []
    states_excl = prefs.get("states_excluded") or []

    if states_pref and "State" in df_scored.columns:
        df_scored = df_scored[df_scored["State"].isin(states_pref)]

    if states_excl and "State" in df_scored.columns:
        df_scored = df_scored[~df_scored["State"].isin(states_excl)]

    # 2. Threshold filters
    thresholds = prefs.get("min_thresholds") or {}
    for col, min_val in thresholds.items():
        if col in df_scored.columns:
            try:
                min_val = float(min_val)
                df_scored = df_scored[df_scored[col] >= min_val]
            except (TypeError, ValueError):
                continue

    # If everything got filtered out, fall back to the original df
    if df_scored.empty:
        df_scored = df_in.copy()

    # 3. Weights
    weights = prefs.get("weights") or {}
    w = {}
    for col in metric_cols:
        try:
            w[col] = float(weights.get(col, 0.0))
        except (TypeError, ValueError):
            w[col] = 0.0

    # If all weights are zero, use equal weights
    if sum(w.values()) == 0:
        w = {col: 1.0 for col in metric_cols}

    total = sum(w.values())
    w = {col: val / total for col, val in w.items()}

    # 4. Compute HawkSight score
    def compute_score(row):
        score = 0.0
        for col in metric_cols:
            if col in row and pd.notna(row[col]):
                try:
                    score += w[col] * float(row[col])
                except (TypeError, ValueError):
                    continue
        return score

    df_scored["hawksight_score"] = df_scored.apply(compute_score, axis=1)
    df_scored = df_scored.sort_values("hawksight_score", ascending=False)

    return df_scored


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

        # Single, elegant search dropdown
        school_list = ["-- Select a school --"] + sorted(df['Institution Name'].unique())
        selected_school_name = st.selectbox("Search for a school by typing its name below:", school_list)

        # Display the profile only when a valid school is selected
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
                    if x <= 1.0:  # convert fractions like 0.56 -> 56.0
                        x *= 100
                    return f"{x:.1f}%"

                if 'Graduation Rate (4yr)' in school and pd.notna(school['Graduation Rate (4yr)']):
                    st.metric(label="4-Year Graduation Rate", value=f"{school['Graduation Rate (4yr)']:.1f}%")

                # 5-year graduation rate
                val = school.get("Graduation Rate (5yr)")
                if pd.isna(val):
                    val = school.get("Graduation rate - Bachelor degree within 5 years  total (DRVGR2023)")
                if pd.notna(val):
                    st.metric(label="5-Year Graduation Rate", value=pct_str(val))

                # 6-year graduation rate
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
        ax.grid(True, linestyle='--', alpha=0.6)

        st.pyplot(fig)
        st.caption("This chart uses a technique called PCA to represent the four complex ranking dimensions on a simple 2D map, revealing the hidden structure in the data.")

    # --- Tab 5: HawkSight Advisor ---
    with tab5:
        st.header("🦅 HawkSight — Precision Guidance for College Decisions")
        st.markdown(
            "Describe the student, and HawkSight will use the same data behind this app to suggest good-fit public colleges "
            "in plain language — based on what the student actually says they care about."
        )

        left_col, right_col = st.columns([1, 1.2])

        with left_col:
            student_description = st.text_area(
                "Who is this student and what do they care about?",
                placeholder=(
                    "Example: I'm a first-generation student interested in biology. "
                    "My family can't afford high tuition, and I care a lot about graduation rates "
                    "and feeling supported on campus."
                ),
                height=140,
            )

            recommend_n = st.selectbox(
                "How many colleges should HawkSight recommend?",
                [2, 3, 4, 5],
                index=1,
            )

            run_button = st.button("Get guidance from HawkSight")

        with right_col:
            st.subheader("HawkSight's Advice")

            if run_button:
                if not student_description.strip():
                    st.warning("Please describe the student first.")
                else:
                    # 1) LLM Call #1: turn text into preferences
                    with st.spinner("Figuring out what matters most to this student..."):
                        prefs = extract_preferences_with_llm(student_description)

                    # 2) Python: apply those preferences to the FULL dataset
                    df_scored = score_schools_with_preferences(df, prefs)

                    # Limit what we SHOW to the LLM (for token reasons),
                    # but note: every school was considered in df_scored.
                    candidate_k = min(40, len(df_scored))
                    candidates = df_scored.head(candidate_k)

                    # Build a compact text chunk for the LLM
                    lines = []
                    for _, row in candidates.iterrows():
                        try:
                            lines.append(
                                f"{row['Institution Name']} ({row['State']}) — "
                                f"Success: {row['student_success_percentile']}, "
                                f"Affordability: {row['affordability_percentile']}, "
                                f"Resources: {row['resources_percentile']}, "
                                f"Equity: {row['equity_percentile']}"
                            )
                        except KeyError:
                            continue

                    context_table = "\n".join(lines)

                    # 3) LLM Call #2: explanation + final recommendations
                    prompt = f"""
Below is a list of public colleges and four scores for each one:
- Student success
- Affordability
- Resources
- Equity

These colleges were selected by matching the student's preferences against the full dataset.
You are HawkSight, a warm, practical advisor for high school students and their families.

Colleges and their scores:
{context_table}

Here is the student:

\"\"\"{student_description}\"\"\"

Your task:
- Recommend **{recommend_n}** colleges from the list above.
- Explain why each recommendation fits this student, in simple, encouraging language.
- Mention trade-offs if helpful (for example, cost vs outcomes or support vs prestige).
- Do NOT add any colleges that are not in the list.
- Do NOT invent extra numeric scores that are not shown.
"""

                    with st.spinner("HawkSight is thinking through options..."):
                        advice = call_hawksight_llm(prompt)

                    st.subheader("HawkSight Recommendation")
                    st.write(advice)
            else:
                st.caption("HawkSight's guidance will appear here after you click the button.")

except Exception as e:
    st.error(f"An unexpected error occurred. Error details: {e}")
