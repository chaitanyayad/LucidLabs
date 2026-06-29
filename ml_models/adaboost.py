import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


def adaboost(X_train, X_test, y_train, y_test, params=None, feature_names=None, task="Classification"):
    if params is None:
        params = {}

    n_estimators = params.get("n_estimators", 50)
    learning_rate = params.get("learning_rate", 1.0)
    max_depth = params.get("max_depth", 1)

    if task == "Classification":
        base = DecisionTreeClassifier(max_depth=max_depth)
        model = AdaBoostClassifier(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("AdaBoost Classifier Results")
        st.metric("Accuracy", f"{score:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            st.pyplot(fig); plt.close()

        with col2:
            if feature_names:
                fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
                fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi) * 0.35)))
                fi.plot(kind="barh", ax=ax2, color="#f59e0b")
                ax2.invert_yaxis(); ax2.set_xlabel("Importance"); ax2.set_title("Feature Importances")
                st.pyplot(fig2); plt.close()

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(2))

    else:
        base = DecisionTreeRegressor(max_depth=max_depth)
        model = AdaBoostRegressor(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("AdaBoost Regressor Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.4f}"); c2.metric("MAE", f"{mae:.4f}"); c3.metric("R²", f"{score:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3, color="#f59e0b")
            mn = min(np.array(y_test).min(), y_pred.min()); mx = max(np.array(y_test).max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted"); st.pyplot(fig); plt.close()

        with col2:
            residuals = np.array(y_test) - y_pred
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3, color="#f59e0b")
            ax2.axhline(0, color="r", linestyle="--"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot"); st.pyplot(fig2); plt.close()

    # Estimator weights over rounds
    st.subheader("Estimator Weights over Boosting Rounds")
    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, len(model.estimator_weights_) + 1), model.estimator_weights_,
             marker="o", markersize=3, color="#f59e0b")
    ax3.set_xlabel("Round"); ax3.set_ylabel("Weight"); ax3.set_title("Estimator Weights")
    st.pyplot(fig3); plt.close()

    st.subheader("How AdaBoost Works")
    st.markdown("""
**AdaBoost (Adaptive Boosting)** trains weak learners sequentially, focusing more on the **misclassified samples** at each round:

1. Initialise uniform sample weights $w_i = 1/n$
2. Train a weak learner (typically a **decision stump** — depth-1 tree)
3. Compute the weighted error rate $\\varepsilon$
4. Assign a higher **estimator weight** $\\alpha = \\frac{1}{2}\\ln\\frac{1-\\varepsilon}{\\varepsilon}$ to accurate learners
5. **Increase** weights of misclassified samples so the next learner focuses on them
6. Final prediction: weighted vote of all weak learners

**Key insight:** By iteratively re-weighting samples, AdaBoost forces each new tree to improve where the current ensemble fails.

**Compared to Gradient Boosting:**
- AdaBoost reweights *samples*; GB fits trees to *residuals*
- AdaBoost is more sensitive to noisy labels / outliers

`max_depth=1` (stumps) is the classic choice — adding depth makes it closer to Gradient Boosting.
    """)

    return score, y_pred
