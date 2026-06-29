import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def linear_regression(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    variant = params.get("variant", "OLS")
    alpha = params.get("alpha", 1.0)

    if variant == "Ridge":
        model = Ridge(alpha=alpha)
    elif variant == "Lasso":
        model = Lasso(alpha=alpha, max_iter=5000)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader("Linear Regression Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
        plt.close()

    with col2:
        residuals = np.array(y_test) - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3)
        ax2.axhline(0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)
        plt.close()

    st.subheader("How Linear Regression Works")
    st.markdown("""
**Linear Regression** fits a hyperplane $\\hat{y} = w^T x + b$ by minimizing **Mean Squared Error**:

$$MSE = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$

Variants:
- **OLS** — plain least squares, no regularization
- **Ridge (L2)** — adds $\\alpha \\|w\\|_2^2$ penalty, shrinks all weights evenly
- **Lasso (L1)** — adds $\\alpha \\|w\\|_1$ penalty, can zero out weights (feature selection)

**R²** (coefficient of determination) measures how much variance is explained by the model. R² = 1.0 is a perfect fit.
    """)

    return r2, y_pred
