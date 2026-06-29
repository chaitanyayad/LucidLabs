import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def support_vector_regressor(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    C = params.get("C", 1.0)
    epsilon = params.get("epsilon", 0.1)
    kernel = params.get("kernel", "rbf")
    gamma = params.get("gamma", "scale")

    model = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Support Vector Regressor Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3, color="darkorange")
        mn = min(np.array(y_test).min(), y_pred.min())
        mx = max(np.array(y_test).max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        plt.close()

    with col2:
        residuals = np.array(y_test) - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3, color="darkorange")
        ax2.axhline(0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)
        plt.close()

    st.subheader("How SVR Works")
    st.markdown("""
**Support Vector Regressor** extends SVM to regression by finding a function that deviates from the actual targets by **at most ε** (epsilon) while being as flat as possible.

The **epsilon-tube** ($\\epsilon$-insensitive zone):
- Predictions within ε of the true value incur **zero loss**
- Only points **outside** the tube (support vectors) contribute to the loss

Key parameters:
- **C** — regularization: large C → fit training data closely (risk overfit); small C → flatter function
- **ε (epsilon)** — width of the tube; larger ε → fewer support vectors, smoother fit
- **Kernel** — same as SVC: `rbf` maps to infinite-dimensional space for non-linear regression

SVR is powerful for small-to-medium datasets with **complex non-linear** relationships and is robust to outliers (unlike linear regression).
    """)

    return r2, y_pred
