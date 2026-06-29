import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def knn_classifier(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    n_neighbors = params.get("n_neighbors", 5)
    metric = params.get("metric", "minkowski")
    weights = params.get("weights", "uniform")

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("K-Nearest Neighbors Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        k_range = range(1, min(21, len(X_train)))
        k_scores = []
        for k in k_range:
            knn_k = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            knn_k.fit(X_train, y_train)
            k_scores.append(accuracy_score(y_test, knn_k.predict(X_test)))
        fig2, ax2 = plt.subplots()
        ax2.plot(list(k_range), k_scores, marker="o")
        ax2.axvline(n_neighbors, color="red", linestyle="--", label=f"k={n_neighbors}")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs k")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("How KNN Works")
    st.markdown("""
**K-Nearest Neighbors** is a **lazy learner** — it memorizes the training set and classifies new points at query time:

1. Compute distance from the query point to all training points
2. Select the **k closest** neighbors
3. Predict by **majority vote** (classification) or average (regression)

**Distance metrics:**
- `minkowski` with p=2 → Euclidean distance (default)
- `manhattan` → sum of absolute differences
- `chebyshev` → max of absolute differences

**Choosing k:**
- Small k → complex boundary, low bias, high variance (overfits)
- Large k → smooth boundary, high bias, low variance (underfits)
- Rule of thumb: try $k = \\sqrt{n}$ where $n$ is the number of training samples

⚠️ KNN is sensitive to **feature scaling** — always normalize features before using it.
    """)

    return acc, y_pred
