import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


def mlp(X_train, X_test, y_train, y_test, params=None, task="Classification"):
    if params is None:
        params = {}

    hidden_layer_sizes = params.get("hidden_layer_sizes", (100,))
    activation = params.get("activation", "relu")
    solver = params.get("solver", "adam")
    alpha = params.get("alpha", 1e-4)
    learning_rate_init = params.get("learning_rate_init", 1e-3)
    max_iter = params.get("max_iter", 300)

    _supports_early_stop = solver in ("sgd", "adam")
    common = dict(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42,
        early_stopping=_supports_early_stop,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )

    if task == "Classification":
        model = MLPClassifier(**common)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("MLP Neural Network (Classifier) Results")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{score:.4f}")
        c2.metric("Epochs trained", model.n_iter_)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="RdPu", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            st.pyplot(fig); plt.close()

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(model.loss_curve_, label="Train loss", color="#a855f7")
            if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
                ax2.plot(model.validation_scores_, label="Val score", color="#6c63ff", linestyle="--")
            ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss / Score")
            ax2.set_title("Training Curve"); ax2.legend()
            st.pyplot(fig2); plt.close()

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(2))

    else:
        model = MLPRegressor(**common)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)

        st.session_state.y_pred = y_pred
        st.session_state.model = model

        st.subheader("MLP Neural Network (Regressor) Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{rmse:.4f}"); c2.metric("MAE", f"{mae:.4f}")
        c3.metric("R²", f"{score:.4f}"); c4.metric("Epochs", model.n_iter_)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3, color="#a855f7")
            mn = min(np.array(y_test).min(), y_pred.min()); mx = max(np.array(y_test).max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted"); st.pyplot(fig); plt.close()

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(model.loss_curve_, color="#a855f7", label="Train loss")
            ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss"); ax2.set_title("Training Curve"); ax2.legend()
            st.pyplot(fig2); plt.close()

    # Architecture summary
    st.subheader("Network Architecture")
    layers = [X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])] + \
             list(hidden_layer_sizes) + \
             [len(np.unique(y_train)) if task == "Classification" else 1]
    arch_str = " → ".join([f"**{l}**" for l in layers])
    st.markdown(f"Input {arch_str} Output")

    st.subheader("How MLP Neural Networks Work")
    st.markdown("""
A **Multi-Layer Perceptron (MLP)** is a fully-connected feedforward neural network:

**Forward pass:**
$$a^{(l)} = \\sigma\\bigl(W^{(l)} a^{(l-1)} + b^{(l)}\\bigr)$$

Each layer applies a linear transformation followed by a non-linear **activation function** $\\sigma$.

**Activation functions:**
- `relu` — $\\max(0, x)$ — most common; avoids vanishing gradients
- `tanh` — maps to $(-1, 1)$; zero-centred
- `logistic` — sigmoid; used for probabilities
- `identity` — no activation (linear MLP)

**Training with backpropagation:**
1. Compute loss (cross-entropy for classification, MSE for regression)
2. Propagate gradients backwards through the network using the chain rule
3. Update weights with an optimiser (`adam`, `sgd`, `lbfgs`)

**Regularisation:**
- `alpha` — L2 weight decay; larger → smaller weights → smoother decision boundary
- `early_stopping` — halts training when validation score stops improving

MLPs can learn **arbitrary non-linear** boundaries but need more data than tree-based methods and are sensitive to **feature scaling** — always standardise inputs.
    """)

    return score, y_pred
