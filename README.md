# Stroke Prediction using Machine Learning

This repository contains a Jupyter Notebook that implements **Machine Learning models** to predict the likelihood of a patient having a stroke based on their health records. The dataset used includes various health parameters such as age, BMI, glucose levels, and lifestyle factors.

## 📌 Features
- **Data Preprocessing**
  - Handles missing values (BMI imputation)
  - Encodes categorical variables
  - Applies Min-Max scaling for numerical features
  
- **Class Balancing**
  - Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle dataset imbalance

- **Model Training and Evaluation**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
  - **Random Forest Classifier**
  - Evaluates models using **Accuracy, Confusion Matrix, Precision, Recall, and F1-score**

## 📂 Files
- `StrokePrediction.ipynb` - Jupyter Notebook containing all implementations
- `healthcare-dataset-stroke-data.csv` - Dataset containing patient health records (ensure it is placed in the working directory)

## 🛠 Dependencies
The notebook uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

To install missing dependencies, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 🔍 How to Use
1. Open the Jupyter Notebook (`StrokePrediction.ipynb`).
2. Run each cell sequentially to preprocess data, balance classes, train models, and evaluate performance.
3. Modify parameters (e.g., test size, solver, scaling method) to experiment with different setups.

## 📈 Results
- **Logistic Regression** provides a baseline performance.
- **SVM** with optimized hyperparameters improves prediction accuracy.
- **Random Forest Classifier** performs best with the highest accuracy and balanced precision-recall.
