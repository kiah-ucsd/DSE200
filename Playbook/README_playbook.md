# 🧠 Machine Learning Workflow Template (v2)

A complete, end-to-end data science pipeline template covering **EDA → Feature Engineering → Train/Test Split → Data Cleaning → Modeling**.  
Designed for reproducibility, interpretability, and clean transitions between analysis and modeling.

---

## ⚙️ Environment

### Prerequisites
- Python ≥ 3.10
- JupyterLab or Jupyter Notebook

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧭 Workflow Overview

| Step | Phase | Description | Output |
|------|--------|-------------|---------|
| **1-3** | Import & Overview | Load data, preview structure, summarize data types & stats | Raw `df` |
| **4-5** | Missing Values & Duplicates | Detect nulls, impute/drop missing data, remove duplicates | Clean base `df` |
| **6-8** | Data Typing & Outliers | Separate numeric/categorical, identify distributions & outliers | Typed and cleaned `df` |
| **9-12** | EDA & Relationships | Explore variable relationships (Numeric ↔ Numeric, Categorical ↔ Numeric, etc.) | Combined `df_combined` |
| **13** | Feature Engineering | Create ratios, deltas, interaction features, group aggregates | Engineered `df_features` |
| **14** | Train-Test Split | Partition engineered dataset into training and test subsets | `X_train`, `X_test`, `y_train`, `y_test` |
| **15** | Data Cleaning / Transformation | Impute → Scale → Validate shapes & distributions | `X_train_scaled`, `X_test_scaled` |
| **16** | Data Export *(optional)* | Save processed datasets to CSV/PKL if needed | `X_train_scaled.csv`, etc. |
| **17** | Modeling & Evaluation | Fit models, assess performance, visualize results | Metrics + plots |

---

## 🧩 Key Features

- Step-by-step modular sections (each cell self-contained)
- Built-in **QA checks** for NaNs, ∞ values, and logical errors
- Integrated **visual sanity checks** (boxplots, heatmaps, pairplots)
- Clear **data lineage**:  
  `raw_df → df_combined → df_features → train/test → scaled → model`
- Ready for extension into pipelines (sklearn Pipeline, MLflow, etc.)

---

## 🧰 Usage

1. Open `ML_Workflow_Template_v2.ipynb` in Jupyter.
2. Follow steps sequentially — each step builds on the previous.
3. (Optional) Modify the feature engineering rules in Step 13 for your dataset.
4. Train models in Step 17; evaluate using accuracy, precision, recall, F1, AUC, etc.

---

## 🧪 Modeling Example (Step 17)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
```

Visualize performance:
```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
```

---

## 💾 Optional: Export Artifacts

Export your trained model and scaler for reuse:
```python
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## ✅ Quality Checklist

- [x] No missing or infinite values after Step 15  
- [x] Numeric distributions normalized  
- [x] Train/test splits stratified  
- [x] Columns consistent between train/test  
- [x] Reproducible environment via `requirements.txt`

---

## 📜 License
MIT License — free to use and modify for research, learning, and production.
