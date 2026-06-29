import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


def gradient_boosting(X_train, X_test, y_train, y_test, params=None, feature_names=None, task="Classification"):
    if params is None:
        params = {}

    n_estimators = params.get("n_estimators", 100)
    learning_rate = params.get("learning_rate", 0.1)
    max_depth = params.get("max_depth", 3)
    subsample = params.get("subsample", 1.0)

    if task == "Classification":
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("Gradient Boosting Classifier Results")
        st.metric("Accuracy", f"{score:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig); plt.close()

        with col2:
            if feature_names:
                fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
                fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi) * 0.35)))
                fi.plot(kind="barh", ax=ax2, color="#6c63ff")
                ax2.invert_yaxis(); ax2.set_xlabel("Importance")
                ax2.set_title("Feature Importances")
                st.pyplot(fig2); plt.close()

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(2))

    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("Gradient Boosting Regressor Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.4f}"); c2.metric("MAE", f"{mae:.4f}"); c3.metric("R²", f"{score:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3, color="#6c63ff")
            mn = min(np.array(y_test).min(), y_pred.min()); mx = max(np.array(y_test).max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted"); st.pyplot(fig); plt.close()

        with col2:
            residuals = np.array(y_test) - y_pred
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3, color="#a855f7")
            ax2.axhline(0, color="r", linestyle="--"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot"); st.pyplot(fig2); plt.close()

    # Learning curve (train loss)
    st.subheader("Training Deviance (Loss over Boosting Rounds)")
    fig3, ax3 = plt.subplots()
    ax3.plot(np.arange(n_estimators) + 1, model.train_score_, label="Train loss", color="#6c63ff")
    ax3.set_xlabel("Boosting rounds"); ax3.set_ylabel("Loss"); ax3.legend()
    st.pyplot(fig3); plt.close()

    st.subheader("How Gradient Boosting Works")
    st.markdown("""
**Gradient Boosting** builds an ensemble of weak learners (shallow trees) **sequentially**:

1. Fit a tree to the **residuals** (errors) of the current ensemble
2. Add the new tree multiplied by the **learning rate** $\\eta$
3. Repeat for `n_estimators` rounds

The key insight: each tree corrects what previous trees got wrong by moving in the direction of the **negative gradient of the loss**.

Key parameters:
- **n_estimators** — number of boosting rounds; more = better fit but slower
- **learning_rate** — shrinks each tree's contribution; smaller η → need more trees but generalises better
- **max_depth** — depth of each individual tree; shallow trees (2–5) are standard
- **subsample** — fraction of training data used per round; < 1.0 introduces stochasticity (Stochastic GB)

Compared to Random Forest: RF trains trees **in parallel** (bagging); GB trains them **sequentially** (boosting).
    """)

    return score, y_pred
