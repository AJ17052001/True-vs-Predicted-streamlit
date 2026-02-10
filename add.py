import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Diabetes Regression Dashboard", layout="wide")

st.title("🩺 Diabetes Progression Predictor")
st.markdown("This dashboard displays the results of a Linear Regression model trained on the Scikit-learn Diabetes dataset.")

# Load Data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Sidebar for interactive parameters
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size (Percentage)", 10, 50, 20) / 100
random_state = st.sidebar.number_input("Random State", value=42)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Metrics in Columns
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("R-squared Score", f"{r2:.2f}")


# Visualization Section
st.subheader("Model Visualizations")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 0: True vs Predicted
axs[0].scatter(y_test, y_pred, color='blue', alpha=0.5)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# Subplot 1: Feature (BMI) vs Predicted
# Index 2 corresponds to BMI in the diabetes dataset
axs[1].scatter(X_test[:, 2], y_pred, color='green', alpha=0.7)
axs[1].set_title("Feature (BMI) vs Predicted Values")
axs[1].set_xlabel("BMI (Feature 2)")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)
