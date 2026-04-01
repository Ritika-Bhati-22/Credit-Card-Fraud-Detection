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

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stAlert {background-color: #e8f4f8;}
    h1 {color: #1f77b4;}
    h2 {color: #2ca02c;}
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
n_samples = st.sidebar.slider("Number of Transactions", 1000, 50000, 10000, 1000)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 30, 5) / 100
fraud_rate = st.sidebar.slider("Fraud Rate (%)", 5, 30, 15, 5) / 100

# Generate data function
@st.cache_data
def generate_fraud_data(n_fraud):
    data = np.zeros((n_fraud, 20))
    data[:, 0] = np.random.exponential(800, n_fraud) + 200  # amount
    data[:, 1] = np.random.choice([0,1,2,3,4,5,22,23], n_fraud)  # time_hour
    data[:, 2] = np.random.randint(0, 7, n_fraud)  # day_of_week
    data[:, 3] = np.random.exponential(500, n_fraud) + 100  # distance_from_home
    data[:, 4] = np.random.exponential(300, n_fraud) + 50  # distance_from_last
    data[:, 5] = np.random.uniform(2.5, 10, n_fraud)  # ratio_to_median
    data[:, 6] = np.random.binomial(1, 0.2, n_fraud)  # repeat_retailer
    data[:, 7] = np.random.binomial(1, 0.3, n_fraud)  # used_chip
    data[:, 8] = np.random.binomial(1, 0.2, n_fraud)  # used_pin
    data[:, 9] = np.random.binomial(1, 0.6, n_fraud)  # online_order
    data[:, 10] = np.random.poisson(3, n_fraud) + 1  # velocity_1h
    data[:, 11] = np.random.poisson(8, n_fraud) + 3  # velocity_24h
    data[:, 12] = data[:, 0] * np.random.uniform(0.8, 1.2, n_fraud)  # avg_last_10
    data[:, 13] = data[:, 12] * np.random.uniform(0.5, 1.5, n_fraud)  # std_last_10
    data[:, 14] = np.random.randint(0, 10, n_fraud)  # merchant_category
    data[:, 15] = np.random.binomial(1, 0.3, n_fraud)  # card_present
    data[:, 16] = np.random.binomial(1, 0.5, n_fraud)  # international
    data[:, 17] = np.random.binomial(1, 0.4, n_fraud)  # high_risk_country
    data[:, 18] = np.random.binomial(1, 0.7, n_fraud)  # unusual_time
    data[:, 19] = (data[:, 2] >= 5).astype(int)  # weekend
    return data

@st.cache_data
def generate_normal_data(n_normal):
    data = np.zeros((n_normal, 20))
    data[:, 0] = np.random.exponential(100, n_normal) + 10
    data[:, 1] = np.random.choice(range(7, 23), n_normal)
    data[:, 2] = np.random.randint(0, 7, n_normal)
    data[:, 3] = np.random.exponential(20, n_normal)
    data[:, 4] = np.random.exponential(15, n_normal)
    data[:, 5] = np.random.uniform(0.5, 2, n_normal)
    data[:, 6] = np.random.binomial(1, 0.7, n_normal)
    data[:, 7] = np.random.binomial(1, 0.8, n_normal)
    data[:, 8] = np.random.binomial(1, 0.7, n_normal)
    data[:, 9] = np.random.binomial(1, 0.3, n_normal)
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
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    fraud_data = generate_fraud_data(n_fraud)
    normal_data = generate_normal_data(n_normal)
    
    data = np.vstack([fraud_data, normal_data])
    labels = np.hstack([np.ones(n_fraud), np.zeros(n_normal)])
    
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    feature_names = [
        'amount', 'time_hour', 'day_of_week', 'distance_from_home',
        'distance_from_last', 'ratio_to_median', 'repeat_retailer', 
        'used_chip', 'used_pin', 'online_order', 'velocity_1h', 
        'velocity_24h', 'avg_last_10', 'std_last_10', 'merchant_category',
        'card_present', 'international', 'high_risk_country', 
        'unusual_time', 'weekend'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    df['is_fraud'] = labels
    
    return df

# Generate data button
if st.sidebar.button("🔄 Generate Data", type="primary"):
    st.session_state['data_generated'] = True
    st.session_state['df'] = generate_dataset(n_samples, fraud_rate)

if 'data_generated' in st.session_state and st.session_state['data_generated']:
    df = st.session_state['df']
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Transactions", f"{len(df):,}")
    with col2:
        fraud_count = int(df['is_fraud'].sum())
        st.metric("🚨 Fraud Cases", f"{fraud_count:,}", 
                 f"{fraud_count/len(df)*100:.1f}%")
    with col3:
        normal_count = len(df) - fraud_count
        st.metric("✅ Normal Cases", f"{normal_count:,}",
                 f"{normal_count/len(df)*100:.1f}%")
    with col4:
        avg_amount = df['amount'].mean()
        st.metric("💰 Avg Amount", f"${avg_amount:.2f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🤖 Model Training", "📈 Results", "🔍 Test Prediction"])
    
    with tab1:
        st.header("Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution
            fraud_dist = df['is_fraud'].value_counts()
            fig = px.pie(values=fraud_dist.values, 
                        names=['Normal', 'Fraud'],
                        title='Transaction Distribution',
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Amount distribution
            fig = px.box(df, x='is_fraud', y='amount', 
                        color='is_fraud',
                        labels={'is_fraud': 'Transaction Type', 'amount': 'Amount ($)'},
                        title='Amount Distribution by Type',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            fig.update_xaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['is_fraud']==0]['time_hour'],
                                      name='Normal', marker_color='#2ecc71',
                                      opacity=0.7, nbinsx=24))
            fig.add_trace(go.Histogram(x=df[df['is_fraud']==1]['time_hour'],
                                      name='Fraud', marker_color='#e74c3c',
                                      opacity=0.7, nbinsx=24))
            fig.update_layout(title='Transaction Time Distribution',
                            xaxis_title='Hour of Day',
                            yaxis_title='Count',
                            barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
            
            # Distance scatter
            fig = px.scatter(df, x='distance_from_home', y='amount',
                           color='is_fraud', 
                           labels={'distance_from_home': 'Distance from Home (km)',
                                  'amount': 'Amount ($)'},
                           title='Distance vs Amount',
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        top_features = ['amount', 'distance_from_home', 'ratio_to_median',
                       'velocity_1h', 'velocity_24h', 'unusual_time', 'is_fraud']
        corr = df[top_features].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Training")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... Please wait"):
                
                # Prepare data
                X = df.drop('is_fraud', axis=1)
                y = df['is_fraud']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
                }
                
                results = {}
                progress_bar = st.progress(0)
                
                for idx, (name, model) in enumerate(models.items()):
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    results[name] = {
                        'model': model,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                    progress_bar.progress((idx + 1) / len(models))
                
                st.session_state['results'] = results
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.success("✅ Models trained successfully!")
    
    with tab3:
        if 'results' in st.session_state:
            st.header("Model Performance")
            
            results = st.session_state['results']
            y_test = st.session_state['y_test']
            
            # Performance comparison
            model_names = list(results.keys())
            accuracies = [results[m]['accuracy'] for m in model_names]
            f1_scores = [results[m]['f1'] for m in model_names]
            roc_aucs = [results[m]['roc_auc'] for m in model_names]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies,
                                    marker_color='#3498db'))
                fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1_scores,
                                    marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='ROC-AUC', x=model_names, y=roc_aucs,
                                    marker_color='#e74c3c'))
                fig.update_layout(title='Model Performance Comparison',
                                barmode='group',
                                yaxis_title='Score',
                                yaxis=dict(range=[0, 1.1]))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROC Curve
                fig = go.Figure()
                for name in model_names:
                    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                           name=f"{name} (AUC={results[name]['roc_auc']:.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                       name='Random', line=dict(dash='dash', color='gray')))
                fig.update_layout(title='ROC Curves',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate')
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(4)
            for idx, name in enumerate(model_names):
                with cols[idx]:
                    cm = results[name]['confusion_matrix']
                    fig = px.imshow(cm, text_auto=True,
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=['Normal', 'Fraud'],
                                   y=['Normal', 'Fraud'],
                                   color_continuous_scale='Blues',
                                   title=name)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance (Random Forest)")
            rf_model = results['Random Forest']['model']
            feature_names = [
                'amount', 'time_hour', 'day_of_week', 'dist_home',
                'dist_last', 'ratio', 'repeat', 'chip', 'pin', 'online',
                'vel_1h', 'vel_24h', 'avg_10', 'std_10', 'category',
                'card_present', 'intl', 'high_risk', 'unusual', 'weekend'
            ]
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            fig = go.Figure(go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                marker_color='#3498db'
            ))
            fig.update_layout(title='Top 10 Important Features',
                            xaxis_title='Importance',
                            yaxis_title='Feature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Please train models first in the 'Model Training' tab")
    
    with tab4:
        if 'results' in st.session_state:
            st.header("Test New Transaction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amount = st.number_input("Amount ($)", 10.0, 10000.0, 100.0)
                time_hour = st.slider("Hour", 0, 23, 12)
                distance_home = st.number_input("Distance from Home (km)", 0.0, 1000.0, 10.0)
                velocity_1h = st.number_input("Transactions in Last Hour", 0, 10, 1)
            
            with col2:
                online = st.selectbox("Online Order", [0, 1], format_func=lambda x: "Yes" if x else "No")
                chip = st.selectbox("Used Chip", [0, 1], format_func=lambda x: "Yes" if x else "No")
                pin = st.selectbox("Used PIN", [0, 1], format_func=lambda x: "Yes" if x else "No")
                international = st.selectbox("International", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            with col3:
                day_of_week = st.slider("Day of Week", 0, 6, 3)
                ratio = st.number_input("Ratio to Median", 0.1, 10.0, 1.0)
                velocity_24h = st.number_input("Transactions in 24h", 0, 20, 3)
                unusual_time = st.selectbox("Unusual Time", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            if st.button("🔍 Predict", type="primary"):
                # Create transaction
                transaction = np.array([[
                    amount, time_hour, day_of_week, distance_home,
                    distance_home * 0.5, ratio, 0, chip, pin, online,
                    velocity_1h, velocity_24h, amount, amount * 0.3, 5,
                    1, international, 0, unusual_time, 1 if day_of_week >= 5 else 0
                ]])
                
                scaler = st.session_state['scaler']
                transaction_scaled = scaler.transform(transaction)
                
                results = st.session_state['results']
                
                st.markdown("### Predictions:")
                
                for name, result in results.items():
                    model = result['model']
                    pred = model.predict(transaction_scaled)[0]
                    proba = model.predict_proba(transaction_scaled)[0]
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        if pred == 1:
                            st.error("🚨 FRAUD")
                        else:
                            st.success("✅ NORMAL")
                    with col3:
                        st.write(f"Confidence: {proba[int(pred)]*100:.1f}%")
        else:
            st.info("👈 Please train models first")

else:
    st.info("👈 Click 'Generate Data' in the sidebar to start!")

