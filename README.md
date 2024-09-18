# Predicting Fraudulent Transactions

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)
11. [Dependencies](#dependencies)
12. [Usage](#usage)
13. [License](#license)

## Introduction
This project aims to predict fraudulent transactions using a machine learning approach. Fraud detection is vital for financial institutions to mitigate losses and protect customers. The project involves data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset
- **Source:** (https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)
- **Description:** The dataset includes various features related to transactions, including amounts, balances, and identifiers.
- **Target Variable:** `isFraud`, indicating whether a transaction is fraudulent.
- **Columns:** Various numerical and categorical features, including transaction amounts, original balances, etc.

## Data Preprocessing
1. **Missing and Duplicate Values:**
   - No missing values or duplicate rows were found in the dataset.
2. **Outlier Detection and Treatment:**
   - Outliers were identified using boxplots.
   - Winsorization was applied to numerical features to cap outliers at 1% on both tails.
3. **Feature Engineering:**
   - Logarithmic features were created to handle skewness in numerical data.
   - Original features like `nameOrig` were dropped to focus on relevant numerical features.

## Exploratory Data Analysis (EDA)
The EDA section involves visualizing the distribution of key features, identifying correlations, and understanding the data's overall structure. Boxplots and histograms were used to visualize outliers and feature distributions.

## Model Building
1. **Feature and Target Selection:**
   - **Features (X):** Processed numerical features.
   - **Target (y):** `isFraud`.
2. **Data Splitting:**
   - Data was split into training (70%) and testing (30%) sets using `train_test_split`.
3. **Handling Class Imbalance:**
   - **SMOTE:** Used to create synthetic samples for the minority class.
   - **Random Under-Sampling:** Reduced the size of the majority class to match the minority class.
4. **Random Forest Classifier:**
   - A Random Forest model was instantiated with 100 estimators.

## Model Evaluation
1. **Accuracy:** 
   - The model achieved an accuracy of **99%** on the test set.
2. **AUC-ROC Score:**
   - The Area Under the ROC Curve (AUC-ROC) score was **0.96**, indicating a high ability to distinguish between fraudulent and non-fraudulent transactions.
3. **Classification Report:**
   - Evaluated the model using metrics like precision, recall, and F1-score.
4. **Precision-Recall Curve:**
   - Visualized the trade-off between precision and recall to identify the optimal threshold.
5. **Optimal Threshold:**
   - Defined an optimal threshold to balance precision and recall.

## Hyperparameter Tuning
1. **Parameter Distribution:**
   - Defined a parameter grid for the Random Forest Classifier, including hyperparameters such as `n_estimators`, `max_features`, `max_depth`, etc.
2. **Randomized Search Cross-Validation:**
   - Performed hyperparameter tuning using `RandomizedSearchCV` with AUC-ROC as the scoring metric.

## Results
- The model showed effective performance in predicting fraudulent transactions with a high accuracy of **99%**.
- The AUC-ROC score of **0.96** demonstrated the model's robustness in distinguishing fraudulent from non-fraudulent transactions.
- The optimal threshold was used to maximize the balance between precision and recall.

## Conclusion
The Random Forest model effectively identified fraudulent transactions, providing a balance between precision and recall. The use of SMOTE and random under-sampling helped address class imbalance, enhancing the model's accuracy.

## Future Work
- Experiment with other machine learning algorithms (e.g., XGBoost, LightGBM).
- Incorporate additional features to improve model performance.
- Explore deep learning techniques for more complex patterns in transaction data.

## Dependencies
- Python (version 3.x)
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (for SMOTE)
- SciPy (for winsorization)

## Usage
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook to execute the analysis:
    ```bash
    jupyter notebook Predicting-Fraudulent-Transactions.ipynb
    ```

## License
This project is licensed under the MIT License.

## Connect with Me
Feel free to connect with me on LinkedIn(https://www.linkedin.com/in/shanttoosh-v-470484289/) and follow my journey in data analytics and visualization!
