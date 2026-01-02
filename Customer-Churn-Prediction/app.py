import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PAGE CONFIG

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})


# HEADER

st.markdown("""
## Customer Churn Prediction  
**Machine Learning Dashboard**
""")

st.markdown("""
This dashboard presents an end-to-end customer churn analysis using supervised machine learning.
It helps identify customers at risk of leaving and the key factors influencing churn.
""")

st.markdown("---")


# LOAD DATA (UNCHANGED PATH)

@st.cache_data
def load_data():
    return pd.read_csv("D:\\Programing World\\Datasets\\Telco-Customer-Churn.csv.csv")

df = load_data()


# METRICS

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Total Customers", df.shape[0])

with m2:
    st.metric("Total Features", df.shape[1])

with m3:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}")


# DATASET PREVIEW

st.markdown("---")
st.markdown("### Dataset Overview")

with st.expander("View Sample Data"):
    st.dataframe(df.head(10), use_container_width=True)


# CHURN DISTRIBUTION

st.markdown("---")
st.markdown("### Churn Distribution")

left, right = st.columns([1, 1])   # right column stays empty

fig, ax = plt.subplots(figsize=(5.5, 4))
sns.countplot(x="Churn", data=df, ax=ax)
ax.set_xlabel("Customer Churn")
ax.set_ylabel("Count")

with left:
    st.pyplot(fig, use_container_width=False)


# CHURN VS TENURE

st.markdown("---")
st.markdown("### Churn vs Customer Tenure")

left, right = st.columns([1, 1])

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax)
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Number of Customers")

with left:
    st.pyplot(fig, use_container_width=False)


# MODELING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model_df = df.copy()
model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})
model_df["TotalCharges"] = pd.to_numeric(model_df["TotalCharges"], errors="coerce")
model_df.dropna(inplace=True)
model_df.drop("customerID", axis=1, inplace=True)

cat_cols = model_df.select_dtypes(include=["object"]).columns
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

X = model_df.drop("Churn", axis=1)
y = model_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)


# CONFUSION MATRIX

st.markdown("---")
st.markdown("### Model Performance")

left, right = st.columns([1, 1])

fig, ax = plt.subplots(figsize=(5.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

with left:
    st.pyplot(fig, use_container_width=False)


# FEATURE IMPORTANCE

st.markdown("---")
st.markdown("### Top Factors Influencing Customer Churn")

left, right = st.columns([1, 1])

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x="Coefficient",
    y="Feature",
    data=feature_importance,
    ax=ax
)

ax.set_xlabel("Impact on Churn")
ax.set_ylabel("Feature")

with left:
    st.pyplot(fig, use_container_width=False)


