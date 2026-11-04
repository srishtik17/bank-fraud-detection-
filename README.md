# 🛡️ Credit Card Fraud Detection

A machine learning project that identifies fraudulent credit card transactions using classification algorithms. This project explores real-world transaction data, handles highly imbalanced classes, performs exploratory data analysis, and compares machine learning models to detect fraudulent activities.

## 📌 Project Overview

Credit card fraud detection is an important application of machine learning in the financial industry. Since fraudulent transactions represent a very small percentage of total transactions, identifying them accurately is a challenging classification problem.

This project uses supervised machine learning techniques to classify transactions as legitimate or fraudulent and compares the performance of Logistic Regression and Random Forest classifiers.

## 📂 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Total Features:** 30 original features + target variable
- **Legitimate Transactions:** 284,315
- **Fraudulent Transactions:** 492
- **Fraud Percentage:** Approximately 0.17%

### Dataset Features

- `Time`: Seconds elapsed between each transaction and the first transaction.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized features obtained using Principal Component Analysis (PCA).
- `Class`: Target variable (0 = Legitimate, 1 = Fraudulent).

## 🛠️ Technologies & Libraries

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Environment:** Jupyter Notebook

### Machine Learning Algorithms
1. Logistic Regression
2. Random Forest Classifier

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Examined dataset structure, dimensions and statistical summaries.
- Checked for missing values.
- Analyzed the distribution of legitimate and fraudulent transactions.
- Visualized transaction amounts using boxplots.
- Explored transaction timing patterns.
- Generated a correlation heatmap to examine relationships between features.
- Compared transaction patterns across fraud and non-fraud classes.

**Key Observation:** The dataset is highly imbalanced, with only 492 fraudulent transactions out of 284,807 records.

## 🔧 Data Preprocessing

The following preprocessing steps were implemented:

1. Loaded the dataset using Pandas.
2. Checked for missing values.
3. Applied `StandardScaler` to the `Amount` and `Time` features.
4. Created two normalized features: `normAmount` and `normTime`.
5. Removed the original `Amount` and `Time` columns after scaling.
6. Created an additional `Hour` feature for time-based analysis.
7. Separated independent features (X) and the target variable (y).
8. Split the dataset into training and testing sets using an 80:20 ratio with stratification.

**Train-Test Split:**
- Training samples: 227,845
- Testing samples: 56,962

## 🤖 Model Building & Evaluation

Two classification algorithms were implemented and evaluated.

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

**Results:**
- Fraud Precision: 83%
- Fraud Recall: 63%
- Fraud F1-Score: 0.72
- ROC-AUC Score: 0.958

**Confusion Matrix:**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,851 | 13 |
| Actual Fraud | 36 | 62 |

### 2. Random Forest Classifier

Random Forest was implemented using an ensemble of decision trees to improve fraud detection performance.

**Configuration:**
- Number of estimators: 30
- Random state: 42
- Parallel processing: Enabled (`n_jobs=-1`)

**Results:**
- Fraud Precision: 94%
- Fraud Recall: 80%
- Fraud F1-Score: 0.86
- ROC-AUC Score: 0.953

**Confusion Matrix:**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,859 | 5 |
| Actual Fraud | 20 | 78 |

## 📈 Model Performance Comparison

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Fraud Precision | 83% | 94% |
| Fraud Recall | 63% | 80% |
| Fraud F1-Score | 0.72 | 0.86 |
| ROC-AUC | 0.958 | 0.953 |
| False Positives | 13 | 5 |
| False Negatives | 36 | 20 |

### Evaluation Techniques

The following evaluation methods were used:

- **Confusion Matrix:** Evaluates correct and incorrect predictions.
- **Classification Report:** Measures precision, recall, F1-score and accuracy.
- **ROC Curve:** Compares the classification performance of both models.
- **Precision-Recall Curve:** Evaluates the trade-off between precision and recall, especially important for imbalanced datasets.
- **Feature Importance:** Identifies influential features in the Random Forest model.

## 💡 Key Insights

- The dataset contains a severe class imbalance, making fraud detection challenging.
- Accuracy alone is not a reliable performance metric for this dataset.
- Random Forest achieved better fraud precision and recall than Logistic Regression.
- Random Forest detected 78 out of 98 fraudulent transactions in the test set.
- Logistic Regression achieved a slightly higher ROC-AUC score.
- Precision, recall, F1-score and confusion matrices provide useful insights into fraud detection performance.

## 💼 Business Impact

An effective credit card fraud detection system can help financial institutions:

- Identify potentially fraudulent transactions.
- Reduce financial losses caused by unauthorized transactions.
- Minimize false fraud alerts.
- Improve customer trust and transaction security.
- Support better risk management and fraud prevention.

## 🔮 Future Enhancements

- Apply SMOTE or other resampling techniques to address class imbalance.
- Experiment with XGBoost and LightGBM.
- Optimize classification thresholds to improve fraud detection.
- Use SHAP for model explainability.
- Improve preprocessing by fitting scalers only on training data.
- Deploy the trained model using FastAPI or Flask.
- Develop a simple web interface for real-time fraud prediction.




## AUTHOR 
Srishti Kumari

**Project Focus:** Exploratory Data Analysis | Data Preprocessing | Imbalanced Classification | Machine Learning | Model Evaluation
