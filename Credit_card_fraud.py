import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colorblind-Safe Dark Theme CSS ───────────────────────────────────────────
# Palette: IBM Colorblind Safe — distinguishable for Deuteranopia, Protanopia,
# Tritanopia. All foreground/background pairs meet WCAG AA (≥4.5:1 contrast).
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Root tokens — IBM Colorblind Safe palette ── */
:root {
    /* backgrounds */
    --bg:        #0a0a0f;
    --surface:   #12131a;
    --surface2:  #1a1b26;
    --border:    #2a2d3e;

    /* IBM colorblind-safe accent set */
    --accent:    #648FFF;   /* blue   — safe anchor colour          */
    --danger:    #FE6100;   /* orange — replaces red (invisible CBP)*/
    --success:   #FFB000;   /* amber  — replaces green (CBP-safe)   */
    --purple:    #785EF0;   /* violet — tertiary accent             */
    --teal:      #009E73;   /* teal   — quaternary (Okabe-Ito safe) */

    /* text */
    --text:      #E8EAF6;
    --muted:     #7B7F9E;

    --radius:    12px;
    --font-body: 'IBM Plex Sans', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* App background */
.stApp {
    background: var(--bg) !important;
}
.main .block-container {
    padding: clamp(1rem, 4vw, 2.5rem) clamp(0.75rem, 3vw, 2rem) !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label { color: var(--text) !important; }

/* ── Title ── */
h1 {
    font-family: var(--font-body) !important;
    font-size: clamp(1.5rem, 4vw, 2.4rem) !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px;
}
h2 {
    font-size: clamp(1.1rem, 3vw, 1.6rem) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}
h3 {
    font-size: clamp(0.95rem, 2.5vw, 1.2rem) !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: clamp(0.75rem, 2vw, 1.25rem) !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent)33 !important;
}
[data-testid="stMetricLabel"] > div { color: var(--muted) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: clamp(1.3rem, 3vw, 1.9rem) !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
    flex-wrap: wrap !important;   /* wrap on small screens */
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-size: clamp(0.75rem, 1.5vw, 0.9rem) !important;
    padding: 0.45rem clamp(0.6rem, 1.5vw, 1rem) !important;
    transition: all .2s !important;
    white-space: nowrap !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
    padding: 0.6rem clamp(1rem, 2.5vw, 1.75rem) !important;
    transition: opacity .2s, transform .15s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Alert boxes ── */
.stAlert {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
}
div[data-testid="stSuccessMessage"] { border-color: var(--success) !important; }
div[data-testid="stErrorMessage"]   { border-color: var(--danger)  !important; }

/* ── Form inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { color: var(--text) !important; }

[data-baseweb="select"] > div,
[data-baseweb="input"]  > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: clamp(0.8rem, 1.5vw, 0.9rem) !important;
}
input[type="number"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [class*="thumb"] { background: var(--accent) !important; }
[data-testid="stSlider"] [class*="track"]:first-child { background: var(--accent) !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }
.stProgress > div { background: var(--border) !important; border-radius: 99px !important; }

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Prediction result cards ── */
.pred-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.pred-model  { flex: 1 1 140px; font-weight: 600; font-size: 0.9rem; }
.pred-label  { flex: 0 0 auto; }
.pred-conf   { flex: 0 0 auto; font-family: var(--font-mono); font-size: 0.85rem; color: var(--muted); }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Mobile overrides (≤ 640 px) ── */
@media (max-width: 640px) {
    /* Stack sidebar controls naturally */
    [data-testid="stSidebar"] { min-width: 0 !important; }

    /* Full-width columns become stacked */
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }

    /* Tighten metric row */
    [data-testid="stMetric"] { padding: 0.65rem 0.75rem !important; }

    /* Plotly charts shouldn't overflow */
    .js-plotly-plot { max-width: 100% !important; }
    .plotly         { max-width: 100% !important; }
}

/* ── Tablet (641-1024 px) ── */
@media (min-width: 641px) and (max-width: 1024px) {
    .main .block-container { padding: 1.5rem 1.25rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
n_samples  = st.sidebar.slider("Number of Transactions", 1000, 50000, 10000, 1000)
test_size  = st.sidebar.slider("Test Size (%)", 10, 40, 30, 5) / 100
fraud_rate = st.sidebar.slider("Fraud Rate (%)", 5, 30, 15, 5) / 100

# ── Data generators ───────────────────────────────────────────────────────────
@st.cache_data
def generate_fraud_data(n_fraud):
    data = np.zeros((n_fraud, 20))
    data[:, 0]  = np.random.exponential(800, n_fraud) + 200
    data[:, 1]  = np.random.choice([0,1,2,3,4,5,22,23], n_fraud)
    data[:, 2]  = np.random.randint(0, 7, n_fraud)
    data[:, 3]  = np.random.exponential(500, n_fraud) + 100
    data[:, 4]  = np.random.exponential(300, n_fraud) + 50
    data[:, 5]  = np.random.uniform(2.5, 10, n_fraud)
    data[:, 6]  = np.random.binomial(1, 0.2, n_fraud)
    data[:, 7]  = np.random.binomial(1, 0.3, n_fraud)
    data[:, 8]  = np.random.binomial(1, 0.2, n_fraud)
    data[:, 9]  = np.random.binomial(1, 0.6, n_fraud)
    data[:, 10] = np.random.poisson(3, n_fraud) + 1
    data[:, 11] = np.random.poisson(8, n_fraud) + 3
    data[:, 12] = data[:, 0] * np.random.uniform(0.8, 1.2, n_fraud)
    data[:, 13] = data[:, 12] * np.random.uniform(0.5, 1.5, n_fraud)
    data[:, 14] = np.random.randint(0, 10, n_fraud)
    data[:, 15] = np.random.binomial(1, 0.3, n_fraud)
    data[:, 16] = np.random.binomial(1, 0.5, n_fraud)
    data[:, 17] = np.random.binomial(1, 0.4, n_fraud)
    data[:, 18] = np.random.binomial(1, 0.7, n_fraud)
    data[:, 19] = (data[:, 2] >= 5).astype(int)
    return data

@st.cache_data
def generate_normal_data(n_normal):
    data = np.zeros((n_normal, 20))
    data[:, 0]  = np.random.exponential(100, n_normal) + 10
    data[:, 1]  = np.random.choice(range(7, 23), n_normal)
    data[:, 2]  = np.random.randint(0, 7, n_normal)
    data[:, 3]  = np.random.exponential(20, n_normal)
    data[:, 4]  = np.random.exponential(15, n_normal)
    data[:, 5]  = np.random.uniform(0.5, 2, n_normal)
    data[:, 6]  = np.random.binomial(1, 0.7, n_normal)
    data[:, 7]  = np.random.binomial(1, 0.8, n_normal)
    data[:, 8]  = np.random.binomial(1, 0.7, n_normal)
    data[:, 9]  = np.random.binomial(1, 0.3, n_normal)
    data[:, 10] = np.random.poisson(1, n_normal)
    data[:, 11] = np.random.poisson(3, n_normal)
    data[:, 12] = data[:, 0] * np.random.uniform(0.9, 1.1, n_normal)
    data[:, 13] = data[:, 12] * np.random.uniform(0.2, 0.5, n_normal)
    data[:, 14] = np.random.randint(0, 10, n_normal)
    data[:, 15] = np.random.binomial(1, 0.8, n_normal)
    data[:, 16] = np.random.binomial(1, 0.1, n_normal)
    data[:, 17] = np.random.binomial(1, 0.05, n_normal)
    data[:, 18] = np.random.binomial(1, 0.2, n_normal)
    data[:, 19] = (data[:, 2] >= 5).astype(int)
    return data

@st.cache_data
def generate_dataset(n_samples, fraud_rate):
    np.random.seed(42)
    n_fraud  = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    fraud_data  = generate_fraud_data(n_fraud)
    normal_data = generate_normal_data(n_normal)
    data   = np.vstack([fraud_data, normal_data])
    labels = np.hstack([np.ones(n_fraud), np.zeros(n_normal)])
    idx    = np.random.permutation(n_samples)
    data, labels = data[idx], labels[idx]
    feature_names = [
        'amount','time_hour','day_of_week','distance_from_home',
        'distance_from_last','ratio_to_median','repeat_retailer',
        'used_chip','used_pin','online_order','velocity_1h',
        'velocity_24h','avg_last_10','std_last_10','merchant_category',
        'card_present','international','high_risk_country',
        'unusual_time','weekend'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['is_fraud'] = labels
    return df

# ── Plotly dark template ───────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#e6edf3', size=11),
    xaxis=dict(gridcolor='#30363d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#30363d', linecolor='#30363d'),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)

def apply_dark(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig

# ── Generate data ─────────────────────────────────────────────────────────────
if st.sidebar.button("🔄 Generate Data", type="primary"):
    st.session_state['data_generated'] = True
    st.session_state['df'] = generate_dataset(n_samples, fraud_rate)

# ══════════════════════════════════════════════════════════════════════════════
if 'data_generated' in st.session_state and st.session_state['data_generated']:
    df = st.session_state['df']

    fraud_count  = int(df['is_fraud'].sum())
    normal_count = len(df) - fraud_count
    avg_amount   = df['amount'].mean()

    # ── KPI row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Total Transactions", f"{len(df):,}")
    c2.metric("🚨 Fraud Cases",  f"{fraud_count:,}",  f"{fraud_count/len(df)*100:.1f}%")
    c3.metric("✅ Normal Cases", f"{normal_count:,}", f"{normal_count/len(df)*100:.1f}%")
    c4.metric("💰 Avg Amount",   f"${avg_amount:.2f}")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration", "🤖 Model Training",
        "📈 Results", "🔍 Test Prediction"
    ])

    # ════════════════════ TAB 1 — Data Exploration ═══════════════════════════
    with tab1:
        st.header("Data Exploration")

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            # Pie
            fig = px.pie(
                values=[normal_count, fraud_count],
                names=['Normal', 'Fraud'],
                title='Transaction Distribution',
                color_discrete_sequence=['#FFB000', '#FE6100'],
                hole=0.4
            )
            apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Box
            fig = px.box(
                df, x='is_fraud', y='amount',
                color='is_fraud',
                labels={'is_fraud': 'Type', 'amount': 'Amount ($)'},
                title='Amount Distribution by Type',
                color_discrete_map={0: '#FFB000', 1: '#FE6100'}
            )
            fig.update_xaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
            apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Time histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud'] == 0]['time_hour'],
                name='Normal', marker_color='#FFB000', opacity=0.75, nbinsx=24
            ))
            fig.add_trace(go.Histogram(
                x=df[df['is_fraud'] == 1]['time_hour'],
                name='Fraud', marker_color='#FE6100', opacity=0.75, nbinsx=24
            ))
            fig.update_layout(
                title='Time-of-Day Distribution',
                xaxis_title='Hour', yaxis_title='Count',
                barmode='overlay', **PLOT_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scatter — sample to keep it fast
            sample = df.sample(min(3000, len(df)), random_state=1)
            fig = px.scatter(
                sample, x='distance_from_home', y='amount',
                color='is_fraud',
                labels={'distance_from_home': 'Dist. from Home (km)', 'amount': 'Amount ($)'},
                title='Distance vs Amount',
                color_discrete_map={0: '#FFB000', 1: '#FE6100'},
                opacity=0.55
            )
            apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap — full width
        st.subheader("Feature Correlation Matrix")
        top_features = [
            'amount', 'distance_from_home', 'ratio_to_median',
            'velocity_1h', 'velocity_24h', 'unusual_time', 'is_fraud'
        ]
        corr = df[top_features].corr()
        fig = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='Cividis',
            title='Correlation Heatmap'
        )
        apply_dark(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════ TAB 2 — Model Training ══════════════════════════════
    with tab2:
        st.header("Model Training")

        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models… this may take a moment"):
                X = df.drop('is_fraud', axis=1)
                y = df['is_fraud']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s  = scaler.transform(X_test)

                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
                }

                results, pbar = {}, st.progress(0)
                for idx, (name, model) in enumerate(models.items()):
                    model.fit(X_train_s, y_train)
                    y_pred  = model.predict(X_test_s)
                    y_proba = model.predict_proba(X_test_s)[:, 1]
                    results[name] = {
                        'model':          model,
                        'predictions':    y_pred,
                        'probabilities':  y_proba,
                        'accuracy':       accuracy_score(y_test, y_pred),
                        'f1':             f1_score(y_test, y_pred),
                        'roc_auc':        roc_auc_score(y_test, y_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                    }
                    pbar.progress((idx + 1) / len(models))

                st.session_state.update({
                    'results': results,
                    'X_test':  X_test_s,
                    'y_test':  y_test,
                    'scaler':  scaler,
                })
                st.success("✅ All models trained successfully!")

    # ════════════════════ TAB 3 — Results ════════════════════════════════════
    with tab3:
        if 'results' in st.session_state:
            st.header("Model Performance")
            results = st.session_state['results']
            y_test  = st.session_state['y_test']

            model_names = list(results.keys())
            accuracies  = [results[m]['accuracy']  for m in model_names]
            f1_scores   = [results[m]['f1']         for m in model_names]
            roc_aucs    = [results[m]['roc_auc']    for m in model_names]

            col1, col2 = st.columns([1, 1], gap="medium")

            with col1:
                fig = go.Figure()
                for vals, name, color in [
                    (accuracies, 'Accuracy',  '#648FFF'),
                    (f1_scores,  'F1-Score',  '#FFB000'),
                    (roc_aucs,   'ROC-AUC',   '#FE6100'),
                ]:
                    fig.add_trace(go.Bar(name=name, x=model_names, y=vals, marker_color=color))
                fig.update_layout(
                    title='Model Performance Comparison', barmode='group',
                    yaxis=dict(range=[0, 1.1], title='Score'), **PLOT_LAYOUT
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                colors = ['#648FFF', '#FFB000', '#FE6100', '#785EF0']
                for name, color in zip(model_names, colors):
                    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=f"{name} (AUC={results[name]['roc_auc']:.3f})",
                        line=dict(color=color, width=2)
                    ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines', name='Random',
                    line=dict(dash='dash', color='#7B7F9E', width=1)
                ))
                fig.update_layout(
                    title='ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    **PLOT_LAYOUT
                )
                st.plotly_chart(fig, use_container_width=True)

            # Confusion matrices — responsive: 2 cols on small, 4 on large
            st.subheader("Confusion Matrices")
            # Use 2 columns per row to be mobile-friendly
            for row_start in range(0, len(model_names), 2):
                cols = st.columns(2, gap="small")
                for i, name in enumerate(model_names[row_start:row_start+2]):
                    cm = results[name]['confusion_matrix']
                    fig = px.imshow(
                        cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                        color_continuous_scale='YlOrBr', title=name
                    )
                    apply_dark(fig)
                    cols[i].plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance (Random Forest)")
            rf_model = results['Random Forest']['model']
            feat_names = [
                'amount','time_hour','day_of_week','dist_home',
                'dist_last','ratio','repeat','chip','pin','online',
                'vel_1h','vel_24h','avg_10','std_10','category',
                'card_present','intl','high_risk','unusual','weekend'
            ]
            importances = rf_model.feature_importances_
            top_idx     = np.argsort(importances)[::-1][:10]
            fig = go.Figure(go.Bar(
                x=importances[top_idx],
                y=[feat_names[i] for i in top_idx],
                orientation='h', marker_color='#648FFF'
            ))
            fig.update_layout(
                title='Top 10 Features', xaxis_title='Importance',
                yaxis=dict(autorange='reversed'), **PLOT_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Please train models first in the **Model Training** tab.")

    # ════════════════════ TAB 4 — Test Prediction ═════════════════════════════
    with tab4:
        if 'results' in st.session_state:
            st.header("Test New Transaction")

            # Responsive 3-column input form
            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                st.markdown("#### 💵 Financial")
                amount        = st.number_input("Amount ($)", 10.0, 10000.0, 100.0)
                ratio         = st.number_input("Ratio to Median", 0.1, 10.0, 1.0)
                velocity_1h   = st.number_input("Transactions / Last Hour", 0, 10, 1)
                velocity_24h  = st.number_input("Transactions / 24 h", 0, 20, 3)

            with col2:
                st.markdown("#### 📍 Location & Time")
                time_hour     = st.slider("Hour of Day", 0, 23, 12)
                day_of_week   = st.slider("Day of Week (0=Mon)", 0, 6, 3)
                distance_home = st.number_input("Distance from Home (km)", 0.0, 1000.0, 10.0)
                unusual_time  = st.selectbox("Unusual Time?", [0, 1], format_func=lambda x: "Yes" if x else "No")

            with col3:
                st.markdown("#### 🔒 Security Flags")
                online        = st.selectbox("Online Order",    [0, 1], format_func=lambda x: "Yes" if x else "No")
                chip          = st.selectbox("Used Chip",       [0, 1], format_func=lambda x: "Yes" if x else "No")
                pin           = st.selectbox("Used PIN",        [0, 1], format_func=lambda x: "Yes" if x else "No")
                international = st.selectbox("International",  [0, 1], format_func=lambda x: "Yes" if x else "No")

            st.markdown("")
            if st.button("🔍 Predict Fraud Risk", type="primary"):
                transaction = np.array([[
                    amount, time_hour, day_of_week, distance_home,
                    distance_home * 0.5, ratio, 0, chip, pin, online,
                    velocity_1h, velocity_24h, amount, amount * 0.3, 5,
                    1, international, 0, unusual_time,
                    1 if day_of_week >= 5 else 0
                ]])
                scaler      = st.session_state['scaler']
                txn_scaled  = scaler.transform(transaction)
                results     = st.session_state['results']

                st.markdown("### 🎯 Predictions")
                for name, result in results.items():
                    pred  = result['model'].predict(txn_scaled)[0]
                    proba = result['model'].predict_proba(txn_scaled)[0]
                    conf  = proba[int(pred)] * 100
                    label = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
                    color = "#FE6100" if pred == 1 else "#FFB000"
                    st.markdown(f"""
                    <div class="pred-card" style="border-color:{color}55;">
                        <span class="pred-model">{name}</span>
                        <span class="pred-label" style="color:{color};font-weight:700;">{label}</span>
                        <span class="pred-conf">Confidence: {conf:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("👈 Please train models first in the **Model Training** tab.")

else:
    # ── Welcome screen ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align:center;
        padding: clamp(2rem,6vw,5rem) 1rem;
        background: #12131a;
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        margin-top: 2rem;
    ">
        <div style="font-size:clamp(2.5rem,8vw,5rem);margin-bottom:1rem;">💳</div>
        <h2 style="color:#648FFF;font-size:clamp(1.2rem,3vw,2rem);margin:0 0 0.75rem;">
            Credit Card Fraud Detection
        </h2>
        <p style="color:#7B7F9E;font-size:clamp(0.85rem,2vw,1.05rem);max-width:520px;margin:0 auto 1.5rem;">
            Generate synthetic transaction data, explore patterns, train ML models,
            and predict fraud in real time — all in one place.
        </p>
        <p style="color:#FFB000;font-size:0.9rem;font-weight:600;">
            ← Click <strong>Generate Data</strong> in the sidebar to begin
        </p>
    </div>
    """, unsafe_allow_html=True)