import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Compare Models", layout="wide")
st.title("Compare Models")
st.markdown(
    "Select multiple models, configure each one, train them all on the same data, "
    "and compare their performance side by side."
)

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

# ── Catalogues ────────────────────────────────────────────────────────────────
CLF_MODELS = {
    "Logistic Regression":  ("sklearn.linear_model",  "LogisticRegression",         {"C": 1.0,             "max_iter": 500}),
    "Random Forest":        ("sklearn.ensemble",       "RandomForestClassifier",     {"n_estimators": 100,  "max_depth": None}),
    "Gradient Boosting":    ("sklearn.ensemble",       "GradientBoostingClassifier", {"n_estimators": 100,  "learning_rate": 0.1}),
    "XGBoost":              ("xgboost",                "XGBClassifier",              {"n_estimators": 100,  "max_depth": 6,    "learning_rate": 0.1}),
    "LightGBM":             ("lightgbm",               "LGBMClassifier",             {"n_estimators": 100,  "num_leaves": 31}),
    "Decision Tree":        ("sklearn.tree",           "DecisionTreeClassifier",     {"max_depth": None}),
    "Extra Trees":          ("sklearn.ensemble",       "ExtraTreesClassifier",       {"n_estimators": 100,  "max_depth": None}),
    "SVM":                  ("sklearn.svm",            "SVC",                        {"C": 1.0,             "kernel": "rbf"}),
    "KNN":                  ("sklearn.neighbors",      "KNeighborsClassifier",       {"n_neighbors": 5}),
    "Naive Bayes":          ("sklearn.naive_bayes",    "GaussianNB",                 {}),
}
REG_MODELS = {
    "Linear Regression":    ("sklearn.linear_model",  "LinearRegression",           {}),
    "Ridge":                ("sklearn.linear_model",  "Ridge",                      {"alpha": 1.0}),
    "Lasso":                ("sklearn.linear_model",  "Lasso",                      {"alpha": 0.1}),
    "ElasticNet":           ("sklearn.linear_model",  "ElasticNet",                 {"alpha": 0.1,         "l1_ratio": 0.5}),
    "Random Forest":        ("sklearn.ensemble",       "RandomForestRegressor",      {"n_estimators": 100,  "max_depth": None}),
    "Gradient Boosting":    ("sklearn.ensemble",       "GradientBoostingRegressor",  {"n_estimators": 100,  "learning_rate": 0.1}),
    "XGBoost":              ("xgboost",                "XGBRegressor",               {"n_estimators": 100,  "max_depth": 6,    "learning_rate": 0.1}),
    "LightGBM":             ("lightgbm",               "LGBMRegressor",              {"n_estimators": 100,  "num_leaves": 31}),
    "SVR":                  ("sklearn.svm",            "SVR",                        {"C": 1.0,             "kernel": "rbf"}),
    "KNN":                  ("sklearn.neighbors",      "KNeighborsRegressor",        {"n_neighbors": 5}),
}

# ── Step 1: Task and data setup ───────────────────────────────────────────────
st.subheader("Step 1 — Configure Data")
task = st.radio("Task type", ["Classification", "Regression"], horizontal=True)
MODEL_CATALOGUE = CLF_MODELS if task == "Classification" else REG_MODELS

cols = df.columns.tolist()
target_col   = st.selectbox("Target column", cols)
feature_cols = st.multiselect(
    "Feature columns",
    [c for c in cols if c != target_col],
    default=[c for c in cols if c != target_col]
)

if not feature_cols:
    st.info("Select at least one feature column.")
    st.stop()

col1, col2, col3 = st.columns(3)
seed      = col1.number_input("Random seed", 0, 9999, 42)
test_size = col2.slider("Test set size (%)", 10, 40, 20) / 100
scale     = col3.checkbox("Scale features (StandardScaler)", value=True)

st.divider()

# ── Step 2: Model selection ───────────────────────────────────────────────────
st.subheader("Step 2 — Select Models to Compare")
selected_models = st.multiselect(
    "Choose models",
    list(MODEL_CATALOGUE.keys()),
    default=list(MODEL_CATALOGUE.keys())[:3]
)

if not selected_models:
    st.info("Select at least one model to continue.")
    st.stop()

# ── Step 3: Configure each model ─────────────────────────────────────────────
st.divider()
st.subheader("Step 3 — Configure Parameters")
st.markdown("Expand each model below to adjust its parameters before training.")

model_configs = {}
for model_name in selected_models:
    module_name, class_name, default_params = MODEL_CATALOGUE[model_name]
    with st.expander(f"{model_name}", expanded=False):
        if not default_params:
            st.caption("No configurable parameters for this model. Using defaults.")
            model_configs[model_name] = {}
            continue

        params = {}
        param_col_list = st.columns(min(len(default_params), 3))
        for i, (pname, default_val) in enumerate(default_params.items()):
            col = param_col_list[i % len(param_col_list)]
            if isinstance(default_val, bool):
                params[pname] = col.checkbox(pname, value=default_val, key=f"{model_name}_{pname}")
            elif isinstance(default_val, int):
                params[pname] = col.number_input(pname, value=default_val, step=1, key=f"{model_name}_{pname}")
            elif isinstance(default_val, float):
                params[pname] = col.number_input(pname, value=default_val, step=0.01, format="%.4f", key=f"{model_name}_{pname}")
            elif isinstance(default_val, str):
                params[pname] = col.text_input(pname, value=default_val, key=f"{model_name}_{pname}")
            elif default_val is None:
                use_limit = col.checkbox(f"Limit {pname}?", value=False, key=f"{model_name}_{pname}_use")
                params[pname] = col.number_input(f"{pname} value", value=5, step=1, key=f"{model_name}_{pname}_val") if use_limit else None
        model_configs[model_name] = params

st.divider()

# ── Step 4: Train and compare ─────────────────────────────────────────────────
st.subheader("Step 4 — Train and Compare")

if st.button("Run Comparison", type="primary"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  precision_score, recall_score,
                                  mean_squared_error, r2_score, mean_absolute_error,
                                  confusion_matrix)

    with st.spinner("Preparing data..."):
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        for c in X.select_dtypes(include=["object", "category"]).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        if task == "Classification":
            le = LabelEncoder()
            y  = le.fit_transform(y.astype(str))
        else:
            y    = pd.to_numeric(y, errors="coerce")
            mask = y.notna()
            X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )

    results   = []
    cms       = {}
    feat_imps = {}
    progress  = st.progress(0, text="Training models...")

    for idx, model_name in enumerate(selected_models):
        progress.progress(idx / len(selected_models), text=f"Training {model_name}...")
        module_name, class_name, _ = MODEL_CATALOGUE[model_name]
        params = model_configs.get(model_name, {})

        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            clean_params = {k: v for k, v in params.items() if v != ""}
            est = cls(**clean_params)

            if scale:
                est = Pipeline([("scaler", StandardScaler()), ("model", est)])

            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)

            if task == "Classification":
                acc       = accuracy_score(y_test, y_pred)
                f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                try:
                    if len(np.unique(y_test)) == 2:
                        proba = est.predict_proba(X_test)[:,1] if hasattr(est, "predict_proba") else y_pred
                        auc   = roc_auc_score(y_test, proba)
                    else:
                        proba = est.predict_proba(X_test) if hasattr(est, "predict_proba") else None
                        auc   = roc_auc_score(y_test, proba, multi_class="ovr") if proba is not None else None
                except Exception:
                    auc = None

                row = {
                    "Model":     model_name,
                    "Accuracy":  round(acc, 4),
                    "Precision": round(precision, 4),
                    "Recall":    round(recall, 4),
                    "F1 Score":  round(f1, 4),
                }
                if auc:
                    row["ROC-AUC"] = round(auc, 4)
                cms[model_name] = confusion_matrix(y_test, y_pred)

            else:
                rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                mae  = round(mean_absolute_error(y_test, y_pred), 4)
                r2   = round(r2_score(y_test, y_pred), 4)
                row  = {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

            row["Status"] = "Success"

            raw = est.named_steps["model"] if scale and hasattr(est, "named_steps") else est
            if hasattr(raw, "feature_importances_"):
                feat_imps[model_name] = pd.Series(raw.feature_importances_, index=feature_cols)
            elif hasattr(raw, "coef_"):
                coef = raw.coef_.flatten()[:len(feature_cols)]
                feat_imps[model_name] = pd.Series(np.abs(coef), index=feature_cols[:len(coef)])

        except Exception as e:
            row = {"Model": model_name, "Status": f"Failed: {str(e)[:80]}"}

        results.append(row)

    progress.progress(1.0, text="All models trained.")
    progress.empty()

    st.session_state["comparison_results"]  = results
    st.session_state["comparison_cms"]      = cms
    st.session_state["comparison_feat_imp"] = feat_imps
    st.session_state["comparison_task"]     = task
    st.session_state["comparison_features"] = feature_cols
    st.session_state["comparison_configs"]  = model_configs
    st.success(f"Trained {len(selected_models)} model(s) successfully.")

# ── Display results ───────────────────────────────────────────────────────────
if "comparison_results" in st.session_state:
    results      = st.session_state["comparison_results"]
    cms          = st.session_state["comparison_cms"]
    feat_imps    = st.session_state["comparison_feat_imp"]
    comp_task    = st.session_state["comparison_task"]
    comp_features = st.session_state.get("comparison_features", feature_cols)

    results_df  = pd.DataFrame(results)
    metric_cols = [c for c in results_df.columns if c not in ("Model", "Status")]

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Metrics Table")
    if metric_cols:
        highlight_max = [c for c in metric_cols if c not in ("RMSE", "MAE")]
        highlight_min = [c for c in metric_cols if c in ("RMSE", "MAE")]
        styled = results_df[["Model"] + metric_cols].style
        if highlight_max:
            styled = styled.highlight_max(axis=0, subset=highlight_max, color="#d4edda")
        if highlight_min:
            styled = styled.highlight_min(axis=0, subset=highlight_min, color="#d4edda")
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(results_df, use_container_width=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if metric_cols:
        st.divider()
        st.subheader("Visual Comparison")
        chosen_metric = st.selectbox("Select metric to visualize", metric_cols)
        plot_df = results_df[results_df[chosen_metric].notna()][["Model", chosen_metric]]

        fig, ax = plt.subplots(figsize=(max(6, len(plot_df) * 1.4), 4))
        vals     = plot_df[chosen_metric].values
        best_idx = vals.argmin() if chosen_metric in ("RMSE", "MAE") else vals.argmax()
        colors   = plt.cm.Blues(np.linspace(0.4, 0.85, len(plot_df)))
        bar_colors = ["#1a6e3c" if i == best_idx else colors[i] for i in range(len(plot_df))]

        bars = ax.bar(plot_df["Model"], vals, color=bar_colors)
        ax.set_title(f"{chosen_metric} by Model  (green = best)")
        ax.set_ylabel(chosen_metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        best_row = plot_df.iloc[best_idx]
        st.success(f"Best model on {chosen_metric}: {best_row['Model']} ({best_row[chosen_metric]:.4f})")

    # ── Radar chart ───────────────────────────────────────────────────────────
    if comp_task == "Classification" and len(results_df) >= 2 and len(metric_cols) >= 2:
        st.divider()
        st.subheader("Multi-Metric Radar Chart")
        radar_df = results_df[["Model"] + metric_cols].dropna()
        if len(radar_df) >= 2:
            N      = len(metric_cols)
            angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            cmap = plt.cm.tab10.colors
            for i, (_, row) in enumerate(radar_df.iterrows()):
                values = [row[c] for c in metric_cols] + [row[metric_cols[0]]]
                ax.plot(angles, values, linewidth=1.5, label=row["Model"], color=cmap[i % 10])
                ax.fill(angles, values, alpha=0.07, color=cmap[i % 10])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_cols, fontsize=10)
            ax.set_title("All metrics comparison", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
            st.pyplot(fig, use_container_width=False)
            plt.close()

    # ── Confusion matrices ────────────────────────────────────────────────────
    if cms:
        st.divider()
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(min(len(cms), 3))
        for i, (model_name, cm) in enumerate(cms.items()):
            with cm_cols[i % 3]:
                fig, ax = plt.subplots(figsize=(3.5, 3))
                ax.imshow(cm, cmap="Blues")
                ax.set_title(model_name, fontsize=10)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Actual", fontsize=8)
                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]):
                        ax.text(c, r, cm[r, c], ha="center", va="center",
                                color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    # ── Feature importance ────────────────────────────────────────────────────
    if feat_imps:
        st.divider()
        st.subheader("Feature Importance Comparison")
        top_n    = st.slider("Top N features to show", 3, min(20, len(comp_features)), min(10, len(comp_features)))
        fi_cols  = st.columns(min(len(feat_imps), 3))
        for i, (model_name, fi) in enumerate(feat_imps.items()):
            with fi_cols[i % 3]:
                top = fi.nlargest(top_n).sort_values()
                fig, ax = plt.subplots(figsize=(4, max(2.5, top_n * 0.35)))
                top.plot(kind="barh", ax=ax, color="#2a6496")
                ax.set_title(model_name, fontsize=10)
                ax.set_xlabel("Importance", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    # ── Export HTML report ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export Report")
    if st.button("Generate HTML Report"):
        metric_headers = "".join(f"<th>{c}</th>" for c in metric_cols)
        html_rows = ""
        for r in results:
            html_rows += f"<tr><td>{r.get('Model','')}</td>"
            for c in metric_cols:
                val = r.get(c, "")
                html_rows += f"<td>{val:.4f}</td>" if isinstance(val, float) else f"<td>{val}</td>"
            html_rows += "</tr>"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ML Studio - Model Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }}
    p {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th {{ background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; }}
    td {{ padding: 8px 14px; border-bottom: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    tr:hover {{ background: #f0f4ff; }}
    .footer {{ margin-top: 40px; font-size: 12px; color: #999; border-top: 1px solid #ddd; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>ML Studio - Model Comparison Report</h1>
  <p>Task: <strong>{comp_task}</strong> &nbsp;|&nbsp; Models evaluated: <strong>{len(results)}</strong></p>
  <table>
    <thead><tr><th>Model</th>{metric_headers}</tr></thead>
    <tbody>{html_rows}</tbody>
  </table>
  <div class="footer">Generated by ML Studio</div>
</body>
</html>"""
        st.download_button("Download HTML Report", html.encode(),
                           "ml_studio_comparison_report.html", "text/html")

    # ── Export predictions ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export Predictions")
    successful = [r["Model"] for r in results if r.get("Status") == "Success"]
    if successful:
        export_model = st.selectbox("Select model to export predictions from", successful)
        if st.button("Export predictions"):
            try:
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.pipeline import Pipeline

                module_name, class_name, _ = MODEL_CATALOGUE[export_model]
                params    = st.session_state.get("comparison_configs", {}).get(export_model, {})
                mod       = __import__(module_name, fromlist=[class_name])
                cls       = getattr(mod, class_name)
                clean_params = {k: v for k, v in params.items() if v != ""}
                est       = cls(**clean_params)
                if scale:
                    est = Pipeline([("scaler", StandardScaler()), ("model", est)])

                X_full = df[comp_features].copy()
                for c in X_full.select_dtypes(include=["object","category"]).columns:
                    X_full[c] = LabelEncoder().fit_transform(X_full[c].astype(str))
                X_full = X_full.fillna(X_full.median(numeric_only=True))

                y_full = df[target_col].copy()
                if comp_task == "Classification":
                    y_full = LabelEncoder().fit_transform(y_full.astype(str))
                else:
                    y_full = pd.to_numeric(y_full, errors="coerce")

                est.fit(X_full, y_full)
                out_df = df.copy()
                out_df["prediction"] = est.predict(X_full)
                csv = out_df.to_csv(index=False).encode()
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not export predictions: {e}")
