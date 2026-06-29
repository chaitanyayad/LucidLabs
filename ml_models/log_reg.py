import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def logistic_regression(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    C = params.get("C", 1.0)
    max_iter = params.get("max_iter", 200)

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Logistic Regression Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        import pandas as pd
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    classes = np.unique(y_test)
    if len(classes) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

    st.subheader("How Logistic Regression Works")
    st.markdown("""
**Logistic Regression** maps inputs to probabilities using the **sigmoid function**:

$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

where $z = w^T x + b$ is a linear combination of features.

- The model learns weights $w$ that minimize **cross-entropy loss**
- Decision boundary: predict class 1 if $\\sigma(z) \\geq 0.5$
- Regularization parameter **C** controls overfitting (smaller C = stronger regularization)
- Works best when classes are **linearly separable** in feature space
    """)

    return acc, y_pred
