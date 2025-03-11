
# 🌽 Corn DON Concentration Prediction  
This repository contains a machine learning project focused on predicting DON (Deoxynivalenol) concentration in corn samples using spectral data. The project includes both a **Random Forest model** and a **Neural Network model** trained and evaluated for performance comparison.

---

## 📂 **Project Structure**  
| File/Folder | Description |
|------------|-------------|
| **app.py** | Streamlit app for predicting DON concentration in new corn samples based on uploaded CSV files. |
| **best_nn_model.h5** | Best-trained neural network model saved in HDF5 format. |
| **comparison_table.png** | A comparison table of performance metrics (MAE, RMSE, R²) between the Random Forest and Neural Network models. |
| **generate_testfile.py** | Python script to generate random test CSV files from the original dataset for testing purposes. |
| **main2.ipynb** | Jupyter notebook containing data preprocessing, model training, evaluation, and visualization. |
| **pca.png** | PCA plot showing dimensionality reduction and variance explained by components. |
| **requirements.txt** | List of dependencies required to run the project. |
| **spectral bands.png** | Visualization of spectral bands used in the dataset. |
| **TASK-ML-INTERN.csv** | Original dataset containing corn samples with spectral bands and DON concentration values. |
| **test_samples_1.csv**, **test_samples_2.csv**, **test_samples_3.csv** | Sample test files generated from the original dataset for prediction testing. |

---

## 🚀 **Setup Instructions**  
### 1. **Clone the Repository**  
```bash
git clone https://github.com/your-username/corn-don-prediction.git
cd corn-don-prediction
```

### 2. **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # For Linux/macOS
.\env\Scripts\activate   # For Windows
```

### 3. **Install Requirements**  
```bash
pip install -r requirements.txt
```

---

## 🎯 **Usage**  
### **1. Streamlit App**  
Run the Streamlit app to make predictions using the trained neural network model:  
```bash
streamlit run app.py
```

- Upload a CSV file containing spectral data.  
- Ensure the first column is labeled `"Sample Name"`.  
- The app will return a downloadable CSV file containing the predictions.  

---

### **2. Generating Test Files**  
Generate sample test files using the script:  
```bash
python generate_testfile.py
```

---

### **3. Training and Evaluation**  
- Open `main2.ipynb` to retrain and evaluate the models.  
- Contains code for data preprocessing, training, hyperparameter tuning, and performance comparison.  

---

## 📊 **Model Performance**  
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Random Forest** | 4205.71 | 12564.48 | 0.4353 |
| **Neural Network** | 3456.04 | 10408.82 | 0.6124 |

---

## 🔥 **Results & Insights**  
✅ The Neural Network model outperformed the Random Forest model in terms of MAE and RMSE, showing better predictive accuracy.  
✅ PCA analysis showed that the dataset's variance is well explained by a few principal components, suggesting high feature relevance.  

---

## 🏆 **Next Steps**  
- Fine-tune hyperparameters to further improve model performance.  
- Explore additional feature engineering techniques.  
- Try other ensemble methods or deep learning architectures.  

---

### ⭐ **If you found this helpful, please give it a star!** 🌟

