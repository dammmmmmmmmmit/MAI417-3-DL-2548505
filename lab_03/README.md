# Lab 03 — Diabetes Classification with Logistic Regression & Regularisation

## About This Program

This lab applies **binary classification** to predict whether a patient has diabetes, using the **Pima Indians Diabetes Dataset**. Multiple regression-based classifiers are compared, including Logistic Regression, Lasso, Ridge, and ElasticNet — all from **scikit-learn** — to study the effect of regularisation techniques on classification performance.

---

## Dataset

| Property     | Detail                                                             |
|--------------|--------------------------------------------------------------------|
| Dataset      | Pima Indians Diabetes (768 samples × 9 features)                  |
| Features     | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age |
| Target       | Outcome: 1 (diabetic), 0 (non-diabetic)                          |
| Class split  | ~35% diabetic, ~65% non-diabetic                                  |

**Note**: The `diabetes.csv` dataset file is excluded from this repository as it is a standard public dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## Models & Approach

1. **Data Loading & EDA** — Loaded with `pandas`, checked shape, data types, class balance
2. **Preprocessing** — `StandardScaler` applied to normalise feature values
3. **Train/Test Split** — `train_test_split` from `sklearn.model_selection`
4. **Models Trained**:
   - **Logistic Regression** — baseline binary classifier
   - **Lasso (L1 regularisation)** — sparsity-inducing; drives irrelevant feature weights toward zero
   - **Ridge (L2 regularisation)** — shrinks all weights; prevents overfitting
   - **ElasticNet (L1 + L2 combined)** — combines benefits of both Lasso and Ridge

---

## Output Interpretation

For each model, the program outputs:

- **Accuracy Score** — overall percentage of correct predictions
- **Confusion Matrix** — shows True Positives, False Positives, True Negatives, False Negatives; useful for understanding class-specific errors
- **Classification Report** — per-class precision, recall, and F1-score

**Key Insight**: Glucose level and BMI tend to be the most predictive features for diabetes onset. Regularised models (Lasso/Ridge) help prevent overfitting on the small dataset (768 rows), typically achieving accuracies in the **75–80% range** on the test set.

Comparing models side-by-side highlights how different regularisation strategies affect the bias-variance trade-off.

---

## Technologies Used

- Python 3
- pandas, NumPy
- scikit-learn (`LogisticRegression`, `Lasso`, `Ridge`, `ElasticNet`, `StandardScaler`, `train_test_split`)
- Matplotlib, Seaborn (visualisation)
