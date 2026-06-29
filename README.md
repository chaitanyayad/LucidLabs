# LucidLabs

An interactive **LucidLabs — ML education platform** built with Streamlit. Users can upload datasets, automatically detect the ML task, explore data, train models, and learn how each algorithm works — all in one app.

---

## What This App Does

This is not a single-script model trainer. It's a full platform that:

- Accepts built-in or user-uploaded datasets
- Automatically detects whether the task is **Classification** or **Regression**
- Runs a mini **EDA (Exploratory Data Analysis)** dashboard
- Cleans and validates data before training
- Lets the user pick an algorithm and trains it
- Shows metrics, visualizations, and **educational explanations** of each algorithm

---

## Project Structure

```
project/
│
├── main.py                  # Orchestrator: UI, routing, session, EDA, dataset loading
│
├── ml_models/               # ML algorithm implementations (called by main.py)
│   ├── log_reg.py           # Logistic Regression
│   ├── lin_reg.py           # Linear Regression
│   ├── ran_for.py           # Random Forest Classifier
│   ├── svm.py               # Support Vector Machine
│   ├── knn.py               # K-Nearest Neighbors
│   ├── decision_tree.py     # Decision Tree
│   └── svr.py               # Support Vector Regressor (for regression tasks)
│
├── assets/                  # Static files
│   ├── styles.css           # App CSS (extracted from inline Python strings)
│   └── landing.html         # Custom welcome/landing page HTML
│
├── pages/                   # (Recommended refactor) Split main.py into page modules
│   ├── eda.py
│   ├── dataset_loader.py
│   └── ui.py
│
├── requirements.txt
└── README.md
```

---

## App Flow

```
Start App
     │
     ▼
Welcome / Landing Page
(Custom HTML + CSS, hides Streamlit branding)
     │
     ▼
Visitor Counter (Redis)
     │
     ▼
[Enter Platform]
     │
     ▼
Select or Upload Dataset
├── Built-in: Iris, Heart Disease, Auto MPG, Concrete Strength
│   (downloaded from UCI repository automatically)
└── Upload: CSV or Excel file
     │
     ▼
Auto Task Detection
├── Target has 0/1 or few unique values → Classification
└── Target has continuous floats → Regression
     │
     ▼
EDA Dashboard
├── Dataset overview & shape
├── Summary statistics
├── Numeric vs Categorical pie chart
├── Missing values report
├── Correlation heatmap
├── Histograms
├── Target distribution analysis
└── Feature summaries
     │
     ▼
Data Cleaning & Validation
(Blocks training if dataset has issues)
     │
     ▼
Feature Selection
     │
     ▼
Choose Algorithm
(List changes based on Classification vs Regression)
     │
     ▼
Train Model
(main.py calls the relevant ml_models/ function)
     │
     ▼
Results
├── Metrics (accuracy, RMSE, R², etc.)
├── Visualizations (confusion matrix, residual plots, etc.)
└── Educational explanation of the chosen algorithm
```

---

## Key Features

### 1. Welcome Page
- Custom HTML landing page with injected CSS
- Streamlit branding hidden
- Redis-based visitor counter
- "Enter Platform" button to proceed

### 2. Session State Management
The app uses `st.session_state` to persist data across tab switches and interactions:
```python
st.session_state.logged_in
st.session_state.data
st.session_state.uploaded_df
st.session_state.y_pred
# ... and more
```
Without this, switching tabs would wipe uploaded datasets.

### 3. Dataset Loading
**Built-in datasets** (auto-downloaded from UCI):
- Iris
- Heart Disease
- Auto MPG
- Concrete Strength

**User uploads:**
- CSV
- Excel (`.xlsx`)

After loading, the app cleans and stores the dataset in session state.

### 4. Automatic Task Detection
```python
if target_column.nunique() <= threshold:
    task = "Classification"
else:
    task = "Regression"
```
This drives which algorithms are shown to the user.

### 5. EDA Module
A comprehensive mini data-analysis dashboard covering:
- Shape, dtypes, head/tail
- `.describe()` summary statistics
- Pie chart: numeric vs categorical features
- Missing value counts and percentages
- Correlation heatmap
- Feature histograms
- Target variable distribution
- Per-feature summaries

### 6. Data Cleaning
Checks for:
- Missing values
- Non-numeric columns that need encoding
- Invalid or empty datasets

Prevents model training until issues are resolved.

### 7. Model Training (Delegation Pattern)
`main.py` does **not** implement ML algorithms. It acts as a controller:

```python
from ml_models.log_reg import logistic_regression
from ml_models.lin_reg import linear_regression
from ml_models.ran_for import random_forest_classifier
# ...

if chosen_model == "Logistic Regression":
    logistic_regression(X_train, X_test, y_train, y_test)
```

Each `ml_models/` file is responsible for:
- Training its model
- Computing metrics
- Returning visualizations
- Displaying results back in Streamlit

### 8. Educational Explanations
For every algorithm, the app explains:
- **SVM**: hyperplane, margin, support vectors (with illustration)
- **Logistic Regression**: sigmoid function, decision boundary
- **Decision Tree**: splitting criteria, depth, leaves
- **Random Forest**: ensemble, bagging, feature importance
- **KNN**: distance metrics, neighborhood voting
- **SVR**: epsilon tube, kernel trick

This makes it a **learning tool**, not just a prediction tool.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| ML Models | scikit-learn |
| Data Handling | pandas, numpy |
| Visualizations | matplotlib, seaborn, plotly |
| Session Persistence | Streamlit session_state |
| Visitor Tracking | Redis |
| Dataset Source | UCI ML Repository |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/lucidlab.git
cd lucidlab

# Install dependencies
pip install -r requirements.txt

# (Optional) Start Redis for visitor counter
redis-server

# Run the app
streamlit run main.py
```

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
redis
openpyxl       # for Excel upload support
ucimlrepo      # for built-in dataset downloads (or use requests)
```

---

## Architecture Notes for Claude Code

> **Read this section carefully before making changes.**

### main.py is the orchestrator
It handles: UI rendering, routing between pages/sections, session state, dataset loading, EDA, and calling ML model functions. It does **not** implement ML algorithms.

### ml_models/ is where algorithms live
Each file exposes one primary function, e.g.:
```python
# ml_models/log_reg.py
def logistic_regression(X_train, X_test, y_train, y_test, params):
    ...
    return metrics, fig
```
When adding a new algorithm, create a new file here and import it in `main.py`.

### Task detection drives algorithm availability
The variable `st.session_state.task` is either `"Classification"` or `"Regression"`. Algorithm selection UI must filter based on this.

### Session state keys (do not rename without updating all references)
```python
st.session_state.logged_in       # bool: whether user has entered the platform
st.session_state.data            # pd.DataFrame: active dataset
st.session_state.uploaded_df     # pd.DataFrame: user-uploaded file
st.session_state.task            # str: "Classification" or "Regression"
st.session_state.target_col      # str: name of the target column
st.session_state.y_pred          # array: model predictions
st.session_state.model           # trained model object
```

### Known areas for improvement
1. `main.py` is very large — consider splitting into `pages/eda.py`, `pages/training.py`, `pages/results.py`
2. Inline CSS/HTML in Python strings should move to `assets/styles.css` and `assets/landing.html`
3. Dataset downloader logic can be extracted to `dataset_loader.py`

---

## Suggested Refactor (Future)

```
main.py           → thin entry point, just routing
pages/
  welcome.py      → landing page + Redis counter
  dataset.py      → dataset loading and upload
  eda.py          → full EDA dashboard
  training.py     → feature selection + model training UI
  results.py      → metrics, plots, explanations
utils/
  session.py      → session state initialization helpers
  task_detect.py  → classification vs regression logic
  cleaner.py      → data validation and cleaning
assets/
  styles.css
  landing.html
ml_models/        → unchanged
```

---

## Project Rating (Self-Assessment)

**~8.5/10**

| Aspect | Notes |
|---|---|
| Modularity | Good — ML logic separated from UI |
| UX | Polished — upload, auto-detection, integrated EDA |
| Educational value | Strong — algorithm explanations built in |
| Maintainability | Could improve — main.py is large, CSS is inline |
| Coverage | Both classification and regression supported |
