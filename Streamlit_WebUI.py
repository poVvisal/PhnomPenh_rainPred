import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data function
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\MSI\Desktop\FN_PJ\NewDATASET\2016-2024-dec.csv") # Replace with actual dataset path
    return data

# App title
st.title("Predictive Modeling App")

# Sidebar
st.sidebar.header("User Inputs")
st.sidebar.markdown("Use the options below to customize the model and input data.")

# Load dataset
data = load_data()

# Drop the 'day' column if it exists
if 'day' in data.columns:
    data = data.drop(columns=['day'])

st.write("## Dataset")
st.dataframe(data.head())

# Preprocess data: Handle categorical columns
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Select feature and target columns
st.sidebar.subheader("Feature Selection")
st.sidebar.markdown("Select the features and target column for the model.")
feature_cols = st.sidebar.multiselect("Select features:", data.columns[:-1], default=data.columns[:-1])
target_col = st.sidebar.selectbox("Select target column:", data.columns[-1:])

# Handle continuous target variable
if data[target_col].dtype in ["float64", "int64"]:
    st.sidebar.write("Detected continuous target variable.")
    st.sidebar.markdown("Adjust the threshold to binarize the target variable.")
    threshold = st.sidebar.slider(
        "Select threshold for binarizing target:",
        min_value=float(data[target_col].min()),
        max_value=float(data[target_col].max()),
        value=float(data[target_col].mean()),
    )
    y = (data[target_col] > threshold).astype(int)
    st.write(f"Target binarized: Values > {threshold} are 1, others are 0.")
else:
    y = data[target_col]

# Prepare the data
X = data[feature_cols]

# Train/Test Split
st.sidebar.subheader("Train/Test Split")
st.sidebar.markdown("Adjust the size of the test set.")
test_size = st.sidebar.slider("Test set size (%):", min_value=10, max_value=50, value=20, step=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

# Display performance metrics
st.write("## Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], "r--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# Confusion Matrix
st.write("## Confusion Matrix")
conf_matrix = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Class 0", "Class 1"])
st.pyplot(conf_matrix.figure_)

# Prediction on user input
st.write("## Make Predictions")
input_data = {col: st.number_input(f"Enter value for {col}:", value=float(data[col].mean())) for col in feature_cols}
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.write(f"**Prediction:** {prediction[0]}")