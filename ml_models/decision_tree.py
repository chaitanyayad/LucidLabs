import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def decision_tree(X_train, X_test, y_train, y_test, params=None, feature_names=None):
    if params is None:
        params = {}

    max_depth = params.get("max_depth", 5)
    criterion = params.get("criterion", "gini")
    min_samples_split = params.get("min_samples_split", 2)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Decision Tree Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax)
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
        fi.plot(kind="barh", ax=ax2, color="saddlebrown")
        ax2.invert_yaxis()
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)
        plt.close()

    depth_cap = min(max_depth if max_depth else 3, 3)
    if depth_cap <= 4:
        st.subheader("Tree Visualization")
        fn = feature_names if feature_names else [f"F{i}" for i in range(X_train.shape[1])]
        classes = [str(c) for c in model.classes_]
        fig3, ax3 = plt.subplots(figsize=(max(12, depth_cap * 4), max(6, depth_cap * 2)))
        plot_tree(model, feature_names=fn, class_names=classes, filled=True, rounded=True,
                  max_depth=depth_cap, ax=ax3, fontsize=8)
        st.pyplot(fig3)
        plt.close()

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("How Decision Trees Work")
    st.markdown("""
**Decision Trees** recursively split the feature space using rules like *"Is feature X > threshold?"*

**Splitting criteria:**
- **Gini impurity**: $G = 1 - \\sum_k p_k^2$ — proportion of misclassified samples if randomly labeled
- **Entropy**: $H = -\\sum_k p_k \\log_2 p_k$ — information gain at each split

The algorithm chooses the split that **maximally reduces impurity**.

**Controlling tree complexity:**
- `max_depth` — hard cap on tree depth; deeper = more complex
- `min_samples_split` — require at least N samples to split a node
- Deeper trees memorize training data → **overfit**; shallower trees → **underfit**

Trees are highly interpretable (you can follow the exact path to any prediction) but prone to high variance — which is why **Random Forest** wraps many of them.
    """)

    return acc, y_pred
