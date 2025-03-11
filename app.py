import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("best_nn_model.h5")

# Load data for scaling
data = pd.read_csv("TASK-ML-INTERN.csv")
X = data.iloc[:, 1:-1].values  # Exclude ID and target column
scaler = StandardScaler()
scaler.fit(X)

# Streamlit app title
st.title("🌽 Corn DON Level Prediction (Hyperspectral Imaging)")

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV file with spectral reflectance values", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        input_data = pd.read_csv(uploaded_file)

        # Check if the CSV has at least two columns (Name + Wavelength data)
        if input_data.shape[1] <= 1:
            st.error("❌ CSV file must contain 449 columns (Name + 448 Spectral Bands columns).")
        else:
            # Extract 'Name' column (first column) and spectral data (remaining columns)
            names = input_data.iloc[:, 0]
            spectral_data = input_data.iloc[:, 1:]

            # Validate column count with model input size
            if spectral_data.shape[1] != X.shape[1]:
                st.error(f"❌ CSV should have exactly {X.shape[1]} spectral columns. Found {spectral_data.shape[1]}.")
            else:
                # Scale the spectral data
                scaled_input = scaler.transform(spectral_data.values)

                # Predict DON levels
                predictions = model.predict(scaled_input)

                # Create result dataframe
                result = pd.DataFrame({
                    "Sample Name": names,
                    "Predicted DON Level (ppb)": predictions.flatten()
                })

                # Display predictions
                st.write("### 🏆 Predicted DON Levels in Corn (ppb):")
                st.write(result)

                # Option to download predictions
                st.download_button(
                    label="📥 Download Predictions",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_don_levels.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# Additional info
st.write("### ℹ️ About")
st.write(
    """
    This app predicts the level of DON (vomitoxin) in corn samples based on hyperspectral imaging data.  
    - Model: Neural Network  
    - Metrics: MAE, RMSE, R²  
    """
)

# Footer
st.write("---")
st.write("Developed by **Vedanta Yadav**")

