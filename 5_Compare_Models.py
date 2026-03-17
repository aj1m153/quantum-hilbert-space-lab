import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Compare Models", layout="wide")
st.title("Compare Models")

# ── Check dataset ─────────────────────────────
if "df" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

# ── Models ───────────────────────────────────
CLF_MODELS = {
    "Logistic Regression": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 500}),
    "Random Forest": ("sklearn.ensemble", "RandomForestClassifier", {}),
    "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingClassifier", {}),
    "SVM": ("sklearn.svm", "SVC", {"probability": True}),
    "KNN": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
}

REG_MODELS = {
    "Linear Regression": ("sklearn.linear_model", "LinearRegression", {}),
    "Ridge": ("sklearn.linear_model", "Ridge", {}),
    "Random Forest": ("sklearn.ensemble", "RandomForestRegressor", {}),
}

# ── Step 1 ───────────────────────────────────
st.subheader("Step 1 — Data Setup")

task = st.radio("Task", ["Classification", "Regression"], horizontal=True)
MODEL_CATALOGUE = CLF_MODELS if task == "Classification" else REG_MODELS

cols = df.columns.tolist()
target = st.selectbox("Target column", cols)
features = st.multiselect("Feature columns", [c for c in cols if c != target],
                         default=[c for c in cols if c != target])

if not features:
    st.stop()

seed = st.number_input("Random seed", 0, 9999, 42)
test_size = st.slider("Test size %", 10, 40, 20) / 100

# ── Step 2 ───────────────────────────────────
st.subheader("Step 2 — Select Models")

selected = st.multiselect(
    "Models",
    list(MODEL_CATALOGUE.keys()),
    default=list(MODEL_CATALOGUE.keys())[:3]
)

if not selected:
    st.stop()

# ── Train ────────────────────────────────────
if st.button("Run Comparison"):

    X = df[features]
    y = df[target]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=int(seed),
        stratify=y if task == "Classification" else None
    )

    results = []
    trained_models = {}

    for model_name in selected:
        module_name, class_name, params = MODEL_CATALOGUE[model_name]

        # Initialize row with ALL metrics (ensures consistency)
        row = {
            "Model": model_name,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC-AUC": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan
        }

        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)

            model = cls(**params)

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            trained_models[model_name] = pipe

            if task == "Classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                row["Accuracy"] = accuracy_score(y_test, preds)
                row["Precision"] = precision_score(y_test, preds, average="weighted", zero_division=0)
                row["Recall"] = recall_score(y_test, preds, average="weighted", zero_division=0)
                row["F1"] = f1_score(y_test, preds, average="weighted")

                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)
                        row["ROC-AUC"] = roc_auc_score(y_test, proba, multi_class="ovr")
                    except:
                        pass

            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                row["RMSE"] = np.sqrt(mean_squared_error(y_test, preds))
                row["MAE"] = mean_absolute_error(y_test, preds)
                row["R2"] = r2_score(y_test, preds)

        except Exception as e:
            row["Error"] = str(e)

        results.append(row)

    st.session_state["results"] = pd.DataFrame(results)
    st.session_state["models"] = trained_models

# ── Results ─────────────────────────────────
if "results" in st.session_state:

    results_df = st.session_state["results"]

    st.subheader("Metrics Table")
    st.dataframe(results_df)

    # Download
    st.download_button("Download Metrics CSV",
                       results_df.to_csv(index=False).encode(),
                       "metrics.csv")

    # ── Visual Comparison ────────────────────
    st.subheader("Visual Comparison")

    metric_cols = [
        c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "RMSE", "MAE", "R2"]
        if c in results_df.columns and results_df[c].notna().any()
    ]

    if metric_cols:
        default_metric = "Precision" if "Precision" in metric_cols else metric_cols[0]

        metric = st.selectbox("Choose metric", metric_cols,
                              index=metric_cols.index(default_metric))

        plot_df = results_df[["Model", metric]].dropna()

        fig, ax = plt.subplots()
        vals = plot_df[metric].values
        best_idx = vals.argmax() if metric not in ["RMSE", "MAE"] else vals.argmin()

        bars = ax.bar(plot_df["Model"], vals)

        for i, b in enumerate(bars):
            if i == best_idx:
                b.set_edgecolor("black")
                b.set_linewidth(2)

        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

    # ── NEW: Precision vs Recall Chart ───────
    if "Precision" in results_df.columns and "Recall" in results_df.columns:

        st.subheader("Precision vs Recall Comparison")

        pr_df = results_df[["Model", "Precision", "Recall"]].dropna()

        if not pr_df.empty:
            x = np.arange(len(pr_df))
            width = 0.35

            fig, ax = plt.subplots()

            ax.bar(x - width/2, pr_df["Precision"], width, label="Precision")
            ax.bar(x + width/2, pr_df["Recall"], width, label="Recall")

            ax.set_xticks(x)
            ax.set_xticklabels(pr_df["Model"], rotation=30)
            ax.set_title("Precision vs Recall")
            ax.legend()

            st.pyplot(fig)

    # ── Export Predictions ───────────────────
    st.subheader("Export Predictions")

    model_choice = st.selectbox("Select model", list(st.session_state["models"].keys()))

    if st.button("Download Predictions"):
        model = st.session_state["models"][model_choice]
        preds = model.predict(df[features])

        out = df.copy()
        out["prediction"] = preds

        st.download_button("Download CSV",
                           out.to_csv(index=False).encode(),
                           "predictions.csv")
