import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# PAGE CONFIG (Responsive)
# ---------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection",
    layout="centered"
)

# ---------------------------------------------------
# RESPONSIVE CSS
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding: 1rem;
}

@media (max-width: 768px) {
    h1 {font-size: 22px;}
    h2 {font-size: 18px;}
}

.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("Credit Card Fraud Detection System")

# ---------------------------------------------------
# MOBILE DETECTION TOGGLE
# ---------------------------------------------------
is_mobile = st.sidebar.checkbox("Mobile Mode", True)

def cols(n):
    if is_mobile:
        return [st.container() for _ in range(n)]
    return st.columns(n)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Configuration")
n_samples = st.sidebar.slider("Transactions", 1000, 50000, 10000)
test_size = st.sidebar.slider("Test Size %", 10, 40, 30) / 100
fraud_rate = st.sidebar.slider("Fraud %", 5, 30, 15) / 100

# ---------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------
@st.cache_data
def generate_data(n, fraud_rate):
    np.random.seed(42)
    fraud = int(n * fraud_rate)
    normal = n - fraud

    def make(f, is_fraud):
        data = np.zeros((f, 6))
        if is_fraud:
            data[:,0] = np.random.exponential(800, f)
            data[:,1] = np.random.exponential(500, f)
            data[:,2] = np.random.uniform(2,10,f)
        else:
            data[:,0] = np.random.exponential(100, f)
            data[:,1] = np.random.exponential(20, f)
            data[:,2] = np.random.uniform(0.5,2,f)

        data[:,3] = np.random.randint(0,24,f)
        data[:,4] = np.random.poisson(3,f)
        data[:,5] = np.random.binomial(1,0.3,f)

        return data

    fraud_data = make(fraud, True)
    normal_data = make(normal, False)

    X = np.vstack([fraud_data, normal_data])
    y = np.hstack([np.ones(fraud), np.zeros(normal)])

    df = pd.DataFrame(X, columns=[
        "amount","distance","ratio","hour","velocity","online"
    ])
    df["is_fraud"] = y

    return df

# ---------------------------------------------------
# GENERATE DATA
# ---------------------------------------------------
if st.sidebar.button("Generate Data"):
    st.session_state.df = generate_data(n_samples, fraud_rate)

if "df" not in st.session_state:
    st.info("Click Generate Data")
    st.stop()

df = st.session_state.df

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
c1,c2,c3 = cols(3)

c1.metric("Total", len(df))
c2.metric("Fraud", int(df.is_fraud.sum()))
c3.metric("Normal", int((df.is_fraud==0).sum()))

st.divider()

# ---------------------------------------------------
# SECTION SELECT (Responsive Tabs)
# ---------------------------------------------------
section = st.selectbox("Select Section", [
    "EDA","Train","Results","Predict"
])

# ---------------------------------------------------
# EDA
# ---------------------------------------------------
if section == "EDA":
    st.subheader("Data Exploration")

    fig = px.pie(df, names="is_fraud")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="is_fraud", y="amount")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
elif section == "Train":
    if st.button("Train Models"):

        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=test_size, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "LR": LogisticRegression(),
            "RF": RandomForestClassifier(),
            "GB": GradientBoostingClassifier(),
            "SVM": SVC(probability=True)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:,1]

            results[name] = {
                "model": model,
                "acc": accuracy_score(y_test,pred),
                "f1": f1_score(y_test,pred),
                "auc": roc_auc_score(y_test,prob)
            }

        st.session_state.results = results
        st.session_state.scaler = scaler
        st.success("Training Done")

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------
elif section == "Results":

    if "results" not in st.session_state:
        st.warning("Train first")
    else:
        results = st.session_state.results

        names = list(results.keys())
        acc = [results[m]["acc"] for m in names]

        fig = go.Figure([go.Bar(x=names,y=acc)])
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif section == "Predict":

    if "results" not in st.session_state:
        st.warning("Train first")
    else:
        c1,c2 = cols(2)

        amount = c1.number_input("Amount",10.0,10000.0,100.0)
        distance = c1.number_input("Distance",0.0,1000.0,10.0)

        ratio = c2.number_input("Ratio",0.1,10.0,1.0)
        velocity = c2.number_input("Velocity",0,20,3)

        if st.button("Predict"):

            x = np.array([[amount,distance,ratio,12,velocity,1]])
            x = st.session_state.scaler.transform(x)

            for name,res in st.session_state.results.items():
                model = res["model"]
                pred = model.predict(x)[0]

                if pred == 1:
                    st.error(f"{name}: FRAUD")
                else:
                    st.success(f"{name}: NORMAL")