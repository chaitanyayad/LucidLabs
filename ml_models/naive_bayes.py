import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def naive_bayes(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {}

    variant = params.get("variant", "Gaussian")
    var_smoothing = params.get("var_smoothing", 1e-9)

    if variant == "Gaussian":
        model = GaussianNB(var_smoothing=var_smoothing)
    elif variant == "Bernoulli":
        model = BernoulliNB(alpha=params.get("alpha", 1.0))
    else:
        # MultinomialNB needs non-negative features — clip to 0
        X_train_nb = np.clip(X_train, 0, None)
        X_test_nb = np.clip(X_test, 0, None)
        model = MultinomialNB(alpha=params.get("alpha", 1.0))
        model.fit(X_train_nb, y_train)
        y_pred = model.predict(X_test_nb)
        _show_results(model, y_pred, y_test, variant)
        return accuracy_score(y_test, y_pred), y_pred

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _show_results(model, y_pred, y_test, variant)
    return accuracy_score(y_test, y_pred), y_pred


def _show_results(model, y_pred, y_test, variant):
    acc = accuracy_score(y_test, y_pred)
    st.session_state.y_pred = y_pred
    st.session_state.model = model

    st.subheader(f"Naive Bayes ({variant}) Results")
    st.metric("Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Per-class prior probabilities
    priors = model.class_prior_
    classes = model.classes_
    fig2, ax2 = plt.subplots()
    ax2.bar([str(c) for c in classes], priors, color="#a855f7")
    ax2.set_xlabel("Class"); ax2.set_ylabel("Prior probability")
    ax2.set_title("Learned Class Priors")
    st.pyplot(fig2); plt.close()

    st.subheader("How Naive Bayes Works")
    st.markdown("""
**Naive Bayes** applies **Bayes' theorem** with the "naive" assumption that features are **conditionally independent** given the class:

$$P(y \\mid x_1, \\ldots, x_n) \\propto P(y) \\prod_{i=1}^n P(x_i \\mid y)$$

The model predicts the class with the highest posterior probability.

**Variants:**
- **Gaussian NB** — assumes $P(x_i \\mid y)$ follows a Gaussian distribution; great for continuous features
- **Multinomial NB** — uses count-based likelihoods; classic choice for text classification (word counts)
- **Bernoulli NB** — binary features (word present/absent); also common in NLP

**Why "naive"?** Real features are almost never independent, yet Naive Bayes works surprisingly well in practice — especially for high-dimensional sparse data like text.

**Pros:** extremely fast to train, needs very little data, interpretable priors.
**Cons:** independence assumption often wrong; can underfit complex boundaries.
    """)
