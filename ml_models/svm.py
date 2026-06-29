import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def support_vector_machine(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    C = params.get("C", 1.0)
    kernel = params.get("kernel", "rbf")
    gamma = params.get("gamma", "scale")

    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Support Vector Machine Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("How SVM Works")
    st.markdown("""
**Support Vector Machine** finds the **maximum-margin hyperplane** that separates classes:

- **Support vectors** are the training points closest to the decision boundary
- **Margin** = distance between support vectors of each class; SVM maximizes this
- **Kernel trick**: maps data to a higher-dimensional space so non-linear boundaries become linear
  - `rbf` (Radial Basis Function) — most common, handles curved boundaries
  - `linear` — fast, good for high-dimensional text data
  - `poly` — polynomial decision boundaries

**C** controls the trade-off:
- Large C → hard margin, low bias, high variance (may overfit)
- Small C → soft margin, more misclassifications allowed, better generalization

**γ (gamma)** controls how far the influence of a single training point reaches.
    """)

    return acc, y_pred
