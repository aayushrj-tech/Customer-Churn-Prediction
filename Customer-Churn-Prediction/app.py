import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

header_left, header_right = st.columns([1, 4])


with header_right:
    st.markdown(
        """
        ## Customer Churn Prediction  
        * Dashboard
        """
    )

st.markdown(
    """
    This dashboard presents an end-to-end customer churn analysis using supervised machine learning.
    It helps identify customers at risk of leaving and the key factors influencing churn.
    """
)

st.markdown("---")


@st.cache_data
def load_data():
    return pd.read_csv("D:\Programing World\Datasets\Telco-Customer-Churn.csv.csv")

df = load_data()

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Total Customers", df.shape[0])

with m2:
    st.metric("Total Features", df.shape[1])

with m3:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}")

st.markdown("---")
st.subheader("Dataset Overview")

with st.expander("View Sample Data"):
    st.dataframe(df.head(10))

st.markdown("---")
st.subheader("Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
ax.set_xticklabels(["No Churn", "Churn"])
ax.set_xlabel("Customer Churn")
ax.set_ylabel("Count")

st.pyplot(fig)


st.markdown("---")
st.subheader("Churn vs Customer Tenure")

fig, ax = plt.subplots()
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax)
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Copy data for modeling
model_df = df.copy()

# Encode target
model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges
model_df["TotalCharges"] = pd.to_numeric(model_df["TotalCharges"], errors="coerce")
model_df = model_df.dropna()

# Drop ID
model_df = model_df.drop("customerID", axis=1)

# Encode categorical features
cat_cols = model_df.select_dtypes(include=["object"]).columns
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

# Split features and target
X = model_df.drop("Churn", axis=1)
y = model_df["Churn"]



#Train Logistic Regression
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

#Confusion Matrix Visualization
st.markdown("---")
st.subheader("Model Performance")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# Feature importance from Logistic Regression
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

st.markdown("---")
st.subheader("Top Factors Influencing Customer Churn")

top_features = feature_importance.head(10)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Coefficient", y="Feature", data=top_features, ax=ax)
ax.set_xlabel("Impact on Churn")
ax.set_ylabel("Feature")

st.pyplot(fig)
