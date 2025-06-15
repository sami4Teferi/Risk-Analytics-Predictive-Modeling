# Insurance Risk Analytics & Predictive Modeling

## Overview

Welcome to the **Insurance Risk Analytics & Predictive Modeling** repository, developed as part of the **10 Academy Artificial Intelligence Mastery challenge (11–17 June 2025)**. This project analyzes historical car insurance claim data (Feb 2014–Aug 2015) for **AlphaCare Insurance Solutions (ACIS)** in South Africa.

**Objectives:**
- Optimize marketing strategies
- Identify low-risk customer segments for competitive premium pricing
- Build predictive models for claim severity and premium optimization

The repository contains Python scripts, Jupyter notebooks, and data pipelines for:

- **Exploratory Data Analysis (EDA):** Summarizes risk and profitability patterns (e.g., Loss Ratio by Province, Gender, VehicleType)
- **Data Version Control (DVC):** Ensures reproducible and auditable data pipelines
- **A/B Hypothesis Testing:** Validates risk differences across provinces, zip codes, and genders
- **Statistical and Machine Learning Models:** Predicts claim severity and optimal premiums using Linear Regression, Random Forest, and XGBoost
- **Model Interpretability:** Uses SHAP to identify key features influencing predictions

This project demonstrates skills in **Data Engineering (DE)**, **Predictive Analytics (PA)**, and **Machine Learning Engineering (MLE)**, simulating real-world financial analytics workflows.

---

## Features

- **EDA:** Descriptive statistics, visualizations, and outlier detection for `TotalPremium`, `TotalClaims`, and categorical variables
- **Data Quality:** Handles missing values, ensures proper data types, and detects anomalies
- **Hypothesis Testing:** Tests null hypotheses on risk and margin differences across provinces, zip codes, and genders
- **Modeling:** Builds models for claim severity (regression) and premium optimization, with feature importance analysis
- **Data Versioning:** Tracks datasets with DVC for regulatory compliance
- **CI/CD:** Uses GitHub Actions for automated testing and linting

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/sami4Teferi/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### Set Up a Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Example `requirements.txt`
```text
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.1
xgboost==2.1.1
shap==0.46.0
jupyter==1.0.0
dvc==3.51.2
scipy==1.13.1
statsmodels==0.14.2
```

### Install DVC and Set Up Remote
```bash
pip install dvc
dvc init
mkdir /path/to/local/storage
dvc remote add -d localstorage /path/to/local/storage
```

### Track Dataset
```bash
dvc add data/insurance_data.csv
dvc push
```

---

## Usage

### Run EDA
Open the Jupyter notebook for Task 1:
```bash
jupyter notebook notebooks/task1_eda.ipynb
```

### Perform Hypothesis Testing
```bash
python scripts/task3_hypothesis_testing.py
```

### Build Models
```bash
jupyter notebook notebooks/task4_modeling.ipynb
```

### View Results
Visualizations and model outputs are saved in the `outputs/` directory.

#### Example Workflow:
1. Preprocess data: `scripts/task1_data_cleaning.py`
2. Perform EDA: `notebooks/task1_eda.ipynb`
3. Run hypothesis tests: `scripts/task3_hypothesis_testing.py`
4. Train models: `notebooks/task4_modeling.ipynb`
5. Review reports: `reports/interim_report.md` and `reports/final_report.md`

---

## Project Structure
```
insurance-risk-analytics/
├── data/                    # Raw and processed datasets (tracked by DVC)
├── notebooks/               # Jupyter notebooks for EDA, hypothesis testing, and modeling
│   ├── task1_eda.ipynb
│   ├── task3_hypothesis.ipynb
│   └── task4_modeling.ipynb
├── scripts/                 # Python scripts for data processing and analysis
│   ├── task1_data_cleaning.py
│   ├── task3_hypothesis_testing.py
│   └── task4_model_training.py
├── outputs/                 # Plots, model results, and visualizations
├── reports/                 # Interim and final reports
│   ├── interim_report.md
│   └── final_report.md
├── .github/                 # GitHub Actions workflows for CI/CD
├── .dvc/                    # DVC configuration
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to ignore in Git
└── README.md                # Project documentation
```

---

## Tasks Completed

### ✅ Task 1: Git & EDA
- Created Git repository with branches (`task-1`, `task-2`, `task-3`, `task-4`)
- Performed EDA on `TotalPremium`, `TotalClaims`, `Province`, `Gender`, and `VehicleType`
- Generated visualizations (e.g., bar charts for Loss Ratio, box plots for outliers)

### ✅ Task 2: DVC
- Initialized DVC and configured local storage
- Tracked dataset (`insurance_data.csv`) with DVC

### ✅ Task 3: A/B Hypothesis Testing
- Tested hypotheses on risk differences (e.g., provinces, zip codes, gender) using chi-squared and t-tests
- Reported p-values and business implications (e.g., regional premium adjustments)

### ✅ Task 4: Statistical Modeling
- Built Linear Regression, Random Forest, and XGBoost models for claim severity and premium prediction
- Evaluated models using RMSE and R-squared
- Used SHAP to identify top features (e.g., vehicle age, province)

---



---

## Contributing

Contributions are welcome!

To contribute:
```bash
# Fork the repository
# Create a new branch
git checkout -b feature/your-feature

# Commit changes
git commit -m "Add your feature"

# Push to GitHub
git push origin feature/your-feature
```

Then, open a Pull Request with a clear description.

✅ Use **Conventional Commits** for commit messages  
✅ Follow **PEP 8** for Python code style

---


---

## Contact

For questions or suggestions:

- Contact **Sami Teferi**
- Or open an issue on [GitHub](https://github.com/sami4Teferi/insurance-risk-analytics)

---


