# 🤖 Machine Learning Classification Project

> **Course:** Wstęp do Uczenia Maszynowego (Introduction to Machine Learning)  
> **Task:** Binary classification on an artificially generated dataset  
> **Metric:** Balanced Accuracy (BA)

---

## 📋 Project Overview

This project proposes and evaluates a classification pipeline designed to maximize **balanced accuracy** on a synthetic dataset with hidden informative features. The dataset contains 100 explanatory variables, many of which may be irrelevant or redundant — a key challenge of this task is identifying and leveraging the truly predictive ones.

The goal is to output **class-1 membership probabilities** for each test observation.

---

## 📁 Repository Structure

```
MachineLearning-Classification_Project/
└── ml_projekt/
    ├── *.ipynb          # Jupyter notebooks with experiments & analysis
    └── *.py             # Supporting Python scripts
```

---

## 📊 Dataset

| Split          | File                          | Size             |
|----------------|-------------------------------|------------------|
| Training data  | `artifical_train_data.csv`    | 1500 observations |
| Training labels| `artifical_train_labels.csv`  | 1500 labels (0/1) |
| Test data      | `artifical_test_data.csv`     | 500 observations  |

- **Features:** 100 numerical variables (some informative, some noise)
- **Task:** Binary classification (class 0 vs class 1)
- **Note:** Data files are not included in this repository

---

## 📐 Evaluation Metric

**Balanced Accuracy (BA)** — accounts for class imbalance by averaging per-class recall:

$$BA = \frac{1}{2} \left( \frac{TP}{P} + \frac{TN}{N} \right)$$

Where:
- $TP/P$ = True Positive Rate (Sensitivity/Recall for class 1)
- $TN/N$ = True Negative Rate (Specificity for class 0)

---

## 🛠️ Methodology

The solution explores the following steps:

1. **Exploratory Data Analysis (EDA)** — distribution of features, class balance, correlation analysis
2. **Feature Selection** — identifying informative variables among 100 candidates (e.g. variance thresholding, mutual information, feature importance from tree models)
3. **Preprocessing** — scaling, handling of low-variance/redundant features
4. **Model Training & Comparison** — multiple classifiers evaluated via cross-validation:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting (XGBoost / LightGBM)
   - Support Vector Machine
5. **Hyperparameter Tuning** — grid/random search with cross-validated balanced accuracy
6. **Final Prediction** — probability outputs for the test set

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm jupyter
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/bartekgit111/MachineLearning-Classification_Project.git
   cd MachineLearning-Classification_Project/ml_projekt
   ```

2. Place the data files in the project directory:
   ```
   artifical_train_data.csv
   artifical_train_labels.csv
   artifical_test_data.csv
   ```

3. Open and run the notebooks in order:
   ```bash
   jupyter notebook
   ```

4. The final prediction file will be saved as:
   ```
   <STUDENT_ID>_artifical_prediction.txt
   ```

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `<ID>_artifical_prediction.txt` | Predicted class-1 probabilities for the test set (500 values) |
| `Kody/` | All source code notebooks and scripts |
| `<ID>_raport.pdf` | Project report (max 4 pages, in Polish) |

---

## 🧪 Results

| Model | CV Balanced Accuracy |
|-------|----------------------|
| Logistic Regression | — |
| Random Forest | — |
| Gradient Boosting | — |
| **Best Model** | **—** |

> Results will be filled in after experiments are complete.

---

## 📚 Technologies Used

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue?logo=pandas)
![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen)

---

## 📝 License

This project is an academic assignment for the *Introduction to Machine Learning* course at **Warsaw University of Technology (WUT)**. Not intended for redistribution.
