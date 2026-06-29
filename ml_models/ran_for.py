import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def random_forest_classifier(X_train, X_test, y_train, y_test, params=None, feature_names=None):
    if params is None:
        params = {}

    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", None)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Random Forest Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi) * 0.35)))
        fi.plot(kind="barh", ax=ax2, color="seagreen")
        ax2.invert_yaxis()
        ax2.set_xlabel("Importance")
        ax2.set_title("Top Feature Importances")
        st.pyplot(fig2)
        plt.close()

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("How Random Forest Works")
    st.markdown("""
**Random Forest** is an **ensemble** of decision trees trained via **bagging**:

1. Draw $B$ bootstrap samples from the training data
2. Fit a decision tree on each sample, but at each split only consider a **random subset of features**
3. Aggregate predictions by majority vote (classification) or averaging (regression)

Key benefits:
- **Reduces variance** compared to a single tree
- **Feature importances** are derived from average impurity decrease across all trees
- Robust to outliers and missing patterns
- `n_estimators` controls the forest size; more trees = more stable but slower
    """)

    return acc, y_pred
