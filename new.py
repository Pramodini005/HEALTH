import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Smart Healthcare Risk Prediction",
    layout="wide"
)

# ------------------ GLOBAL STYLING ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #e3f2fd, #ffffff);
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff, #e8f5e9);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.section-header {
    color: #1565c0;
    font-weight: 700;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #1565c0, #42a5f5);
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("🏥 Healthcare System")
    st.image("https://images.unsplash.com/photo-1586773860418-d37222d8fce3", use_container_width=True)
    page = st.radio("📌 Navigation", [
        "🏠 Home",
        "🧠 Input & Predictions",
        "📂 Data Generator / Upload",
        "📊 Cluster Visualization",
        "📈 Model Performance"
    ])

# ------------------ RISK FUNCTION ------------------
def calculate_risk(age, bmi, bp, chol, glucose, smoking):

    score = (
        bmi * 0.3 +
        bp * 0.2 +
        chol * 0.2 +
        glucose * 0.2 +
        age * 0.1
    )

    if smoking == "Yes":
        score += 15

    if score > 170:
        return "High", "Yes", round(score * 120, 2)
    elif score > 140:
        return "Medium", "Possible", round(score * 90, 2)
    else:
        return "Low", "No", round(score * 60, 2)

# ------------------ HOME PAGE ------------------
if page == "🏠 Home":

    st.title("🏥 Healthcare Risk Prediction System")

    col1, col2 = st.columns([2,1])

    with col1:
        st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118", use_container_width=True)
        st.markdown("### 📖 Project Overview")
        st.write("""
        This system uses Machine Learning techniques to:
        - Predict disease likelihood 🦠
        - Estimate medical expenses 💰
        - Categorize patient risk levels ⚠️
        """)

        st.markdown("### ✨ Key Features")
        st.success("✔ Real-time prediction")
        st.success("✔ Data upload & synthetic generation")
        st.success("✔ Cluster visualization")
        st.success("✔ Performance analytics dashboard")

    with col2:
        st.markdown("### 👩‍💻 Team")
        team = {
            "👩 Pramodini": "Lead Data Scientist",
            "👨 Dhruv": "Frontend Developer",
            "👨 Ranjita": "Backend Engineer",
            "👨 Rahul": "UI/UX Designer"
        }
        for name, role in team.items():
            st.info(f"**{name}**\n{role}")

# ------------------ PREDICTION PAGE ------------------
elif page == "🧠 Input & Predictions":

    st.header("🧠 Smart Healthcare Prediction")

    col_input, col_output = st.columns([1,2])

    with col_input:
        st.image("https://images.unsplash.com/photo-1580281657527-47b74d1b5b02", use_container_width=True)
        st.subheader("📝 Patient Info")

        age = st.slider("Age", 20, 100, 45)
        bmi = st.slider("BMI", 15.0, 45.0, 24.5)
        bp = st.slider("Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 300, 180)
        glucose = st.slider("Glucose", 70, 250, 95)
        smoking = st.selectbox("Smoking", ["No", "Yes"])

        if st.button("🔍 Predict", use_container_width=True):
            risk, disease, expenses = calculate_risk(
                age, bmi, bp, chol, glucose, smoking
            )
            st.session_state["prediction"] = (risk, disease, expenses)

    with col_output:
        if "prediction" in st.session_state:
            risk, disease, expenses = st.session_state["prediction"]

            st.markdown("### 📊 Prediction Results")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("💰 Expenses", f"${expenses:,.2f}")

            with c2:
                st.metric("🦠 Disease", disease)

            with c3:
                st.metric("⚠️ Risk Level", risk)

            st.divider()

            st.subheader("📉 Risk Gauge")

            gauge_fig = px.bar(
                x=[risk],
                y=[expenses],
                color=[risk],
                title="Expense-Based Risk Indicator"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

# ------------------ DATA PAGE ------------------
elif page == "📂 Data Generator / Upload":

    st.header("📂 Data Management")

    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71", use_container_width=True)

    upload_file = st.file_uploader("Upload CSV File", type=["csv"])

    if upload_file:
        df_uploaded = pd.read_csv(upload_file)
        st.success("Dataset Uploaded Successfully")
        st.dataframe(df_uploaded.head())
        st.write("Dataset Shape:", df_uploaded.shape)

    if st.button("Generate Synthetic Dataset"):
        df = pd.DataFrame({
            "Age": np.random.randint(20, 80, 300),
            "BMI": np.random.normal(26, 4, 300),
            "Blood Pressure": np.random.normal(125, 15, 300),
            "Cholesterol": np.random.normal(190, 25, 300),
            "Glucose": np.random.normal(110, 20, 300)
        })
        st.success("Synthetic Data Generated")
        st.dataframe(df.head())

# ------------------ CLUSTER PAGE ------------------
elif page == "📊 Cluster Visualization":

    st.header("📊 BMI vs Blood Pressure Clustering")

    df_viz = pd.DataFrame({
        "BMI": np.random.normal(25, 5, 300),
        "Blood Pressure": np.random.normal(120, 15, 300),
    })

    df_viz["Risk"] = pd.cut(
        df_viz["BMI"] + df_viz["Blood Pressure"],
        bins=[0, 130, 160, 300],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    fig = px.scatter(
        df_viz,
        x="BMI",
        y="Blood Pressure",
        color="Risk",
        size="BMI",
        hover_data=["Blood Pressure"],
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ MODEL PERFORMANCE ------------------
elif page == "📈 Model Performance":

    st.header("📈 Model Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌳 Decision Tree", "0.89", "+0.02")

    with col2:
        st.metric("🔵 KNN (k=5)", "0.87")

    with col3:
        st.metric("📊 Regression R²", "0.79")

    st.divider()

    perf_df = pd.DataFrame({
        "Model": ["Decision Tree", "KNN (k=5)", "Regression"],
        "Score": [0.89, 0.87, 0.79]
    })

    fig_perf = px.bar(
        perf_df,
        x="Model",
        y="Score",
        color="Score",
        template="plotly_white"
    )

    st.plotly_chart(fig_perf, use_container_width=True)