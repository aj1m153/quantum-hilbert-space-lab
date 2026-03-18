import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Compare Models", layout="wide")
st.title("Compare Models")
st.markdown(
    "Select multiple models, configure each one with feature engineering and tuning, "
    "train them all on the same data, and compare every metric side by side."
)

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def encode_and_fill(X):
    from sklearn.preprocessing import LabelEncoder
    for c in X.select_dtypes(include=["object","category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X.fillna(X.median(numeric_only=True))

def apply_fe(X, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs):
    X = X.copy()
    if "Log transform" in fe_options:
        for c in log_cols:
            if c in X.columns:
                X[f"log_{c}"] = np.log1p(X[c].clip(lower=0))
    if "Binning" in fe_options:
        for c in bin_cols:
            if c in X.columns:
                X[f"bin_{c}"] = pd.cut(X[c], bins=bin_bins, labels=False)
    if "Interaction terms" in fe_options:
        for c1, c2 in interact_pairs:
            if c1 in X.columns and c2 in X.columns:
                X[f"{c1}_x_{c2}"] = X[c1] * X[c2]
    if "Polynomial features" in fe_options and poly_degree > 1:
        from sklearn.preprocessing import PolynomialFeatures
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        arr  = poly.fit_transform(X[num_cols])
        names = poly.get_feature_names_out(num_cols)
        poly_df = pd.DataFrame(arr, columns=names, index=X.index)
        X = pd.concat([X.drop(columns=num_cols), poly_df], axis=1)
    return X.fillna(0)

# ═════════════════════════════════════════════════════════════════════════════
# CATALOGUES
# ═════════════════════════════════════════════════════════════════════════════
CLF_MODELS = {
    "Logistic Regression": ("sklearn.linear_model",  "LogisticRegression",         {"C": 1.0,            "max_iter": 500}),
    "Random Forest":       ("sklearn.ensemble",       "RandomForestClassifier",     {"n_estimators": 100, "max_depth": None}),
    "Gradient Boosting":   ("sklearn.ensemble",       "GradientBoostingClassifier", {"n_estimators": 100, "learning_rate": 0.1}),
    "XGBoost":             ("xgboost",                "XGBClassifier",              {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}),
    "LightGBM":            ("lightgbm",               "LGBMClassifier",             {"n_estimators": 100, "num_leaves": 31}),
    "Decision Tree":       ("sklearn.tree",           "DecisionTreeClassifier",     {"max_depth": None}),
    "Extra Trees":         ("sklearn.ensemble",       "ExtraTreesClassifier",       {"n_estimators": 100, "max_depth": None}),
    "SVM":                 ("sklearn.svm",            "SVC",                        {"C": 1.0,            "kernel": "rbf"}),
    "KNN":                 ("sklearn.neighbors",      "KNeighborsClassifier",       {"n_neighbors": 5}),
    "Naive Bayes":         ("sklearn.naive_bayes",    "GaussianNB",                 {}),
}
REG_MODELS = {
    "Linear Regression":  ("sklearn.linear_model", "LinearRegression",           {}),
    "Ridge":              ("sklearn.linear_model", "Ridge",                      {"alpha": 1.0}),
    "Lasso":              ("sklearn.linear_model", "Lasso",                      {"alpha": 0.1}),
    "ElasticNet":         ("sklearn.linear_model", "ElasticNet",                 {"alpha": 0.1, "l1_ratio": 0.5}),
    "Random Forest":      ("sklearn.ensemble",     "RandomForestRegressor",      {"n_estimators": 100, "max_depth": None}),
    "Gradient Boosting":  ("sklearn.ensemble",     "GradientBoostingRegressor",  {"n_estimators": 100, "learning_rate": 0.1}),
    "XGBoost":            ("xgboost",              "XGBRegressor",               {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}),
    "LightGBM":           ("lightgbm",             "LGBMRegressor",              {"n_estimators": 100, "num_leaves": 31}),
    "SVR":                ("sklearn.svm",          "SVR",                        {"C": 1.0, "kernel": "rbf"}),
    "KNN":                ("sklearn.neighbors",    "KNeighborsRegressor",        {"n_neighbors": 5}),
}

CLF_GRID = {
    "Logistic Regression": {"C":[0.01,0.1,1,10],           "max_iter":[500]},
    "Random Forest":       {"n_estimators":[50,100,200],   "max_depth":[None,5,10]},
    "Gradient Boosting":   {"n_estimators":[50,100],       "learning_rate":[0.05,0.1,0.2]},
    "XGBoost":             {"n_estimators":[50,100],       "max_depth":[3,6], "learning_rate":[0.05,0.1]},
    "LightGBM":            {"n_estimators":[50,100],       "num_leaves":[31,63]},
    "Decision Tree":       {"max_depth":[None,5,10,20]},
    "Extra Trees":         {"n_estimators":[50,100],       "max_depth":[None,5,10]},
    "SVM":                 {"C":[0.1,1,10],                "kernel":["rbf","linear"]},
    "KNN":                 {"n_neighbors":[3,5,7,11]},
    "Naive Bayes":         {},
}
REG_GRID = {
    "Linear Regression":  {},
    "Ridge":              {"alpha":[0.1,1,10,100]},
    "Lasso":              {"alpha":[0.01,0.1,1,10]},
    "ElasticNet":         {"alpha":[0.1,1], "l1_ratio":[0.2,0.5,0.8]},
    "Random Forest":      {"n_estimators":[50,100,200], "max_depth":[None,5,10]},
    "Gradient Boosting":  {"n_estimators":[50,100],     "learning_rate":[0.05,0.1]},
    "XGBoost":            {"n_estimators":[50,100],     "max_depth":[3,6], "learning_rate":[0.05,0.1]},
    "LightGBM":           {"n_estimators":[50,100],     "num_leaves":[31,63]},
    "SVR":                {"C":[0.1,1,10],              "kernel":["rbf","linear"]},
    "KNN":                {"n_neighbors":[3,5,7,11]},
}

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols     = df.columns.tolist()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("Step 1 — Data Configuration", expanded=True):
    task = st.radio("Task type", ["Classification", "Regression"], horizontal=True, key="cmp_task")
    MODEL_CATALOGUE = CLF_MODELS if task == "Classification" else REG_MODELS
    GRID_CATALOGUE  = CLF_GRID   if task == "Classification" else REG_GRID

    target_col   = st.selectbox("Target column", all_cols, key="cmp_target")
    feature_cols = st.multiselect("Feature columns",
                                   [c for c in all_cols if c != target_col],
                                   default=[c for c in all_cols if c != target_col],
                                   key="cmp_features")
    if not feature_cols:
        st.info("Select at least one feature column.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    seed      = c1.number_input("Random seed", 0, 9999, 42, key="cmp_seed")
    test_size = c2.slider("Test set size (%)", 10, 40, 20, key="cmp_split") / 100
    scale     = c3.checkbox("Scale features (StandardScaler)", value=True, key="cmp_scale")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING (shared across all models)
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("Step 2 — Feature Engineering (applied to all models)", expanded=False):
    fe_options = st.multiselect("Select transformations",
        ["Log transform","Binning","Interaction terms","Polynomial features"], key="cmp_fe")
    log_cols, bin_cols, bin_bins, poly_degree, interact_pairs = [], [], 4, 2, []
    if "Log transform" in fe_options:
        log_cols = st.multiselect("Columns to log-transform", numeric_cols, key="cmp_log")
    if "Binning" in fe_options:
        bin_cols = st.multiselect("Columns to bin", numeric_cols, key="cmp_bin")
        bin_bins = st.slider("Number of bins", 2, 20, 4, key="cmp_bins")
    if "Interaction terms" in fe_options:
        n_pairs = st.number_input("Number of interaction pairs", 1, 10, 1, key="cmp_npairs")
        for i in range(int(n_pairs)):
            cc1, cc2 = st.columns(2)
            a = cc1.selectbox(f"Pair {i+1} — column A", numeric_cols, key=f"cmp_ia_{i}")
            b = cc2.selectbox(f"Pair {i+1} — column B", numeric_cols, key=f"cmp_ib_{i}")
            interact_pairs.append((a, b))
    if "Polynomial features" in fe_options:
        poly_degree = st.slider("Polynomial degree", 2, 4, 2, key="cmp_poly")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — SELECT & CONFIGURE MODELS
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("Step 3 — Select and Configure Models", expanded=True):
    selected_models = st.multiselect("Choose models to compare",
                                      list(MODEL_CATALOGUE.keys()),
                                      default=list(MODEL_CATALOGUE.keys())[:3],
                                      key="cmp_selected")
    if not selected_models:
        st.info("Select at least one model.")
        st.stop()

    model_configs = {}
    for model_name in selected_models:
        module_name, class_name, default_params = MODEL_CATALOGUE[model_name]
        default_grid = GRID_CATALOGUE[model_name]
        st.markdown(f"**{model_name}**")
        c_tune, c_params = st.columns([1, 3])
        tuning = c_tune.radio("Tuning", ["Manual","GridSearch","Optuna"],
                               horizontal=False, key=f"cmp_tune_{model_name}")
        params   = {}
        gs_grid  = default_grid
        n_trials = 20

        if tuning == "Manual" and default_params:
            param_cols = c_params.columns(min(len(default_params), 4))
            for i, (pname, default_val) in enumerate(default_params.items()):
                col = param_cols[i % len(param_cols)]
                if isinstance(default_val, bool):
                    params[pname] = col.checkbox(pname, value=default_val, key=f"cmp_{model_name}_{pname}")
                elif isinstance(default_val, int):
                    params[pname] = col.number_input(pname, value=default_val, step=1, key=f"cmp_{model_name}_{pname}")
                elif isinstance(default_val, float):
                    params[pname] = col.number_input(pname, value=default_val, step=0.01, format="%.4f", key=f"cmp_{model_name}_{pname}")
                elif isinstance(default_val, str):
                    params[pname] = col.text_input(pname, value=default_val, key=f"cmp_{model_name}_{pname}")
                elif default_val is None:
                    use_lim = col.checkbox(f"Limit {pname}?", value=False, key=f"cmp_{model_name}_{pname}_use")
                    params[pname] = col.number_input(f"{pname} value", value=5, step=1, key=f"cmp_{model_name}_{pname}_val") if use_lim else None
        elif tuning == "GridSearch" and default_grid:
            import json
            grid_str = c_params.text_area(f"Grid JSON", value=json.dumps(default_grid, indent=2),
                                           height=100, key=f"cmp_grid_{model_name}")
            try:    gs_grid = json.loads(grid_str)
            except: gs_grid = default_grid
        elif tuning == "Optuna":
            n_trials = c_params.number_input("Trials", 5, 100, 20, key=f"cmp_trials_{model_name}")

        model_configs[model_name] = {
            "module": module_name, "class": class_name,
            "params": params, "tuning": tuning,
            "gs_grid": gs_grid, "n_trials": n_trials,
            "default_grid": default_grid
        }
        st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — RUN COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
if st.button("Run Comparison", type="primary", key="cmp_run"):
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  precision_score, recall_score,
                                  confusion_matrix, classification_report,
                                  roc_curve, precision_recall_curve,
                                  mean_squared_error, r2_score, mean_absolute_error)

    with st.spinner("Preparing data..."):
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        X = encode_and_fill(X)
        X = apply_fe(X, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs)

        if task == "Classification":
            le = LabelEncoder()
            y  = le.fit_transform(y.astype(str))
        else:
            y    = pd.to_numeric(y, errors="coerce")
            mask = y.notna(); X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )

    all_results  = []
    all_cms      = {}
    all_feat_imp = {}
    all_probs    = {}

    progress = st.progress(0, text="Training models...")

    for idx, model_name in enumerate(selected_models):
        progress.progress(idx / len(selected_models), text=f"Training {model_name}...")
        cfg = model_configs[model_name]

        try:
            mod = __import__(cfg["module"], fromlist=[cfg["class"]])
            cls = getattr(mod, cfg["class"])
            clean_params = {k: v for k, v in cfg["params"].items() if v != ""}

            if cfg["tuning"] == "Manual":
                est = cls(**clean_params)
            elif cfg["tuning"] == "GridSearch" and cfg["gs_grid"]:
                base = cls(random_state=int(seed)) if "random_state" in cls.__init__.__code__.co_varnames else cls()
                cv   = GridSearchCV(base, cfg["gs_grid"], cv=3, n_jobs=-1,
                                     scoring="accuracy" if task=="Classification" else "neg_root_mean_squared_error")
                cv.fit(X_train, y_train)
                est = cv.best_estimator_
            elif cfg["tuning"] == "Optuna" and cfg["default_grid"]:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                def objective(trial):
                    p = {k: trial.suggest_categorical(k, v) for k, v in cfg["default_grid"].items()}
                    try:
                        m = cls(**p)
                        return cross_val_score(m, X_train, y_train, cv=3,
                            scoring="accuracy" if task=="Classification" else "neg_root_mean_squared_error").mean()
                    except: return -9999
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=int(cfg["n_trials"]))
                est = cls(**study.best_params)
            else:
                est = cls(**clean_params)

            if scale:
                est = Pipeline([("scaler", StandardScaler()), ("model", est)])

            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)

            row = {"Model": model_name, "Status": "Success"}

            if task == "Classification":
                is_binary = len(np.unique(y_test)) == 2
                acc   = accuracy_score(y_test, y_pred)
                prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                auc   = None
                try:
                    if is_binary:
                        y_prob = est.predict_proba(X_test)[:,1] if hasattr(est,"predict_proba") else None
                    else:
                        y_prob = est.predict_proba(X_test) if hasattr(est,"predict_proba") else None
                    if y_prob is not None:
                        auc = roc_auc_score(y_test, y_prob, multi_class="ovr" if not is_binary else "raise")
                        all_probs[model_name] = (y_prob, is_binary)
                except: pass

                ks_stat = None
                if auc and is_binary and y_prob is not None:
                    from scipy import stats as sp
                    ks_stat, _ = sp.ks_2samp(y_prob[y_test==1], y_prob[y_test==0])
                    somers_d   = 2 * auc - 1

                row.update({"Accuracy": round(acc,4), "Precision": round(prec,4),
                             "Recall": round(rec,4), "F1 Score": round(f1,4)})
                if auc:      row["ROC-AUC"]  = round(auc, 4)
                if ks_stat:  row["KS Stat"]  = round(ks_stat, 4)
                if auc:      row["Somers D"] = round(2*auc-1, 4)
                all_cms[model_name] = confusion_matrix(y_test, y_pred)

            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test.values - y_pred) /
                       np.where(y_test.values==0,1,y_test.values))) * 100
                row.update({"RMSE": round(rmse,4), "MAE": round(mae,4),
                             "R2": round(r2,4), "MAPE": round(mape,2)})

            # Feature importance
            raw = est.named_steps["model"] if scale and hasattr(est,"named_steps") else est
            feat_names = list(X_train.columns)
            if hasattr(raw, "feature_importances_"):
                all_feat_imp[model_name] = pd.Series(raw.feature_importances_, index=feat_names)
            elif hasattr(raw, "coef_"):
                coef = raw.coef_.flatten()[:len(feat_names)]
                all_feat_imp[model_name] = pd.Series(np.abs(coef), index=feat_names[:len(coef)])

        except Exception as e:
            row = {"Model": model_name, "Status": f"Failed: {str(e)[:80]}"}

        all_results.append(row)

    progress.progress(1.0, text="All models trained.")
    progress.empty()

    # Store in session state
    st.session_state["cmp_results"]   = all_results
    st.session_state["cmp_cms"]       = all_cms
    st.session_state["cmp_feat_imp"]  = all_feat_imp
    st.session_state["cmp_probs"]     = all_probs
    st.session_state["cmp_task"]      = task
    st.session_state["cmp_y_test"]    = y_test
    st.session_state["cmp_features"]  = list(X_train.columns)
    st.success(f"Trained {len(selected_models)} model(s) successfully.")

# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═════════════════════════════════════════════════════════════════════════════
if "cmp_results" not in st.session_state:
    st.stop()

all_results  = st.session_state["cmp_results"]
all_cms      = st.session_state["cmp_cms"]
all_feat_imp = st.session_state["cmp_feat_imp"]
all_probs    = st.session_state["cmp_probs"]
comp_task    = st.session_state["cmp_task"]
y_test       = st.session_state["cmp_y_test"]
comp_features = st.session_state["cmp_features"]

results_df  = pd.DataFrame(all_results)
metric_cols = [c for c in results_df.columns if c not in ("Model","Status")]
lower_better = ["RMSE","MAE","MAPE"]
higher_better = [c for c in metric_cols if c not in lower_better]

st.divider()

# ── Metrics table ─────────────────────────────────────────────────────────────
with st.expander("Metrics Table", expanded=True):
    if metric_cols:
        styled = results_df[["Model"] + metric_cols].style
        if higher_better:
            styled = styled.highlight_max(axis=0, subset=higher_better, color="#d4edda")
        if lower_better and any(c in metric_cols for c in lower_better):
            styled = styled.highlight_min(axis=0,
                         subset=[c for c in lower_better if c in metric_cols], color="#d4edda")
        st.dataframe(styled, use_container_width=True)
        st.caption("Green = best value for each metric")
    else:
        st.dataframe(results_df, use_container_width=True)

# ── Bar chart comparison ───────────────────────────────────────────────────────
with st.expander("Visual Comparison", expanded=True):
    if metric_cols:
        chosen_metric = st.selectbox("Select metric to visualize", metric_cols, key="cmp_viz_metric")
        plot_df = results_df[results_df[chosen_metric].notna()][["Model", chosen_metric]]
        fig, ax = plt.subplots(figsize=(max(6, len(plot_df)*1.4), 4))
        vals     = plot_df[chosen_metric].values
        best_idx = vals.argmin() if chosen_metric in lower_better else vals.argmax()
        colors   = plt.cm.Blues(np.linspace(0.4, 0.85, len(plot_df)))
        bar_colors = ["#1a6e3c" if i == best_idx else colors[i] for i in range(len(plot_df))]
        bars = ax.bar(plot_df["Model"], vals, color=bar_colors)
        ax.set_title(f"{chosen_metric} by Model  (green = best)")
        ax.set_ylabel(chosen_metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.001,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        best_row = plot_df.iloc[best_idx]
        st.success(f"Best model on {chosen_metric}: {best_row['Model']} ({best_row[chosen_metric]:.4f})")

# ── Radar chart ───────────────────────────────────────────────────────────────
if comp_task == "Classification" and len(results_df) >= 2 and len(metric_cols) >= 2:
    with st.expander("Radar Chart — All Metrics", expanded=False):
        radar_df = results_df[["Model"] + metric_cols].dropna()
        if len(radar_df) >= 2:
            N      = len(metric_cols)
            angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
            fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
            cmap = plt.cm.tab10.colors
            for i, (_, row) in enumerate(radar_df.iterrows()):
                values = [row[c] for c in metric_cols] + [row[metric_cols[0]]]
                ax.plot(angles, values, linewidth=1.5, label=row["Model"], color=cmap[i%10])
                ax.fill(angles, values, alpha=0.07, color=cmap[i%10])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_cols, fontsize=9)
            ax.set_title("All metrics — model comparison", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
            st.pyplot(fig, use_container_width=False); plt.close()

# ── Confusion matrices ────────────────────────────────────────────────────────
if all_cms:
    with st.expander("Confusion Matrices", expanded=False):
        cm_cols = st.columns(min(len(all_cms), 3))
        for i, (model_name, cm) in enumerate(all_cms.items()):
            with cm_cols[i % 3]:
                fig, ax = plt.subplots(figsize=(3.5, 3))
                ax.imshow(cm, cmap="Blues")
                ax.set_title(model_name, fontsize=10)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Actual", fontsize=8)
                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]):
                        ax.text(c, r, cm[r,c], ha="center", va="center",
                                color="white" if cm[r,c] > cm.max()/2 else "black", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

# ── ROC curves (classification, binary) ──────────────────────────────────────
if all_probs and comp_task == "Classification":
    binary_probs = {k: v for k, v in all_probs.items() if v[1]}
    if binary_probs:
        with st.expander("ROC-AUC Curves", expanded=False):
            from sklearn.metrics import roc_curve, auc as sk_auc
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap = plt.cm.tab10.colors
            for i, (model_name, (y_prob, _)) in enumerate(binary_probs.items()):
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_val = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc_val:.4f})", color=cmap[i%10])
            ax.plot([0,1],[0,1],"k--", lw=1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves — All Models"); ax.legend(fontsize=9); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("Precision-Recall Curves", expanded=False):
            from sklearn.metrics import precision_recall_curve
            fig, ax = plt.subplots(figsize=(6, 5))
            for i, (model_name, (y_prob, _)) in enumerate(binary_probs.items()):
                prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
                ax.plot(rec_c, prec_c, lw=2, label=model_name, color=cmap[i%10])
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curves — All Models"); ax.legend(fontsize=9); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("Lift and Cumulative Gains Curves", expanded=False):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for i, (model_name, (y_prob, _)) in enumerate(binary_probs.items()):
                sorted_idx    = np.argsort(y_prob)[::-1]
                sorted_labels = np.array(y_test)[sorted_idx]
                pct_pop  = np.arange(1, len(sorted_labels)+1) / len(sorted_labels)
                pct_pos  = np.cumsum(sorted_labels) / sorted_labels.sum()
                lift     = pct_pos / pct_pop
                axes[0].plot(pct_pop*100, lift, lw=2, label=model_name, color=cmap[i%10])
                axes[1].plot(pct_pop*100, pct_pos*100, lw=2, label=model_name, color=cmap[i%10])
            axes[0].axhline(1, color="gray", linestyle="--", lw=1, label="Baseline")
            axes[0].set_xlabel("% Population"); axes[0].set_ylabel("Lift")
            axes[0].set_title("Lift Curves"); axes[0].legend(fontsize=8)
            axes[1].plot([0,100],[0,100],"k--",lw=1,label="Baseline")
            axes[1].set_xlabel("% Population"); axes[1].set_ylabel("% Positives Captured")
            axes[1].set_title("Cumulative Gains"); axes[1].legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("KS Plots", expanded=False):
            ks_cols = st.columns(min(len(binary_probs), 3))
            for i, (model_name, (y_prob, _)) in enumerate(binary_probs.items()):
                thresholds = np.linspace(0, 1, 100)
                tpr_l, fpr_l = [], []
                for t in thresholds:
                    pred_t = (y_prob >= t).astype(int)
                    tp = ((pred_t==1)&(y_test==1)).sum()
                    fp = ((pred_t==1)&(y_test==0)).sum()
                    fn = ((pred_t==0)&(y_test==1)).sum()
                    tn = ((pred_t==0)&(y_test==0)).sum()
                    tpr_l.append(tp/(tp+fn) if (tp+fn)>0 else 0)
                    fpr_l.append(fp/(fp+tn) if (fp+tn)>0 else 0)
                ks_diff = np.abs(np.array(tpr_l) - np.array(fpr_l))
                ks_max  = np.argmax(ks_diff)
                with ks_cols[i % 3]:
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.plot(thresholds, tpr_l, label="TPR")
                    ax.plot(thresholds, fpr_l, label="FPR")
                    ax.axvline(thresholds[ks_max], color="red", linestyle="--",
                               label=f"KS={ks_diff[ks_max]:.3f}")
                    ax.set_title(model_name, fontsize=10)
                    ax.set_xlabel("Threshold"); ax.legend(fontsize=8); plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close()

# ── Feature importance comparison ─────────────────────────────────────────────
if all_feat_imp:
    with st.expander("Feature Importance Comparison", expanded=False):
        top_n   = st.slider("Top N features", 3, min(20, len(comp_features)), min(10, len(comp_features)), key="cmp_topn")
        fi_cols = st.columns(min(len(all_feat_imp), 3))
        for i, (model_name, fi) in enumerate(all_feat_imp.items()):
            with fi_cols[i % 3]:
                top = fi.nlargest(top_n).sort_values()
                fig, ax = plt.subplots(figsize=(4, max(2.5, top_n*0.35)))
                top.plot(kind="barh", ax=ax, color="#2a6496")
                ax.set_title(model_name, fontsize=10)
                ax.set_xlabel("Importance", fontsize=8); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

# ── Regression: residual comparison ──────────────────────────────────────────
if comp_task == "Regression" and "cmp_y_test" in st.session_state:
    with st.expander("Regression Diagnostic Summary", expanded=False):
        st.info("Run individual models on the Train Model page for full diagnostics (VIF, Breusch-Pagan, QQ plots). This section shows a high-level metric summary.")
        if metric_cols:
            fig, axes = plt.subplots(1, len([c for c in ["RMSE","MAE","R2","MAPE"] if c in metric_cols]),
                                      figsize=(14, 4))
            if not isinstance(axes, np.ndarray): axes = [axes]
            plot_metrics = [c for c in ["RMSE","MAE","R2","MAPE"] if c in metric_cols]
            for ax, metric in zip(axes, plot_metrics):
                plot_df = results_df[results_df[metric].notna()][["Model", metric]]
                colors  = plt.cm.Blues(np.linspace(0.4, 0.85, len(plot_df)))
                ax.bar(plot_df["Model"], plot_df[metric], color=colors)
                ax.set_title(metric); ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

# ── Export HTML report ────────────────────────────────────────────────────────
with st.expander("Export Report", expanded=False):
    if st.button("Generate HTML Report", key="cmp_html"):
        metric_headers = "".join(f"<th>{c}</th>" for c in metric_cols)
        html_rows = ""
        for r in all_results:
            html_rows += f"<tr><td>{r.get('Model','')}</td>"
            for c in metric_cols:
                val = r.get(c,"")
                html_rows += f"<td>{val:.4f}</td>" if isinstance(val,float) else f"<td>{val}</td>"
            html_rows += "</tr>"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ML Studio - Model Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }}
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
  <p>Task: <strong>{comp_task}</strong> | Models evaluated: <strong>{len(all_results)}</strong></p>
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
    successful = [r["Model"] for r in all_results if r.get("Status") == "Success"]
    if successful and "df" in st.session_state:
        export_model = st.selectbox("Export predictions from model", successful, key="cmp_export_sel")
        if st.button("Export predictions CSV", key="cmp_export_btn"):
            try:
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.pipeline import Pipeline
                cfg = model_configs[export_model]
                mod = __import__(cfg["module"], fromlist=[cfg["class"]])
                cls = getattr(mod, cfg["class"])
                clean_params = {k: v for k, v in cfg["params"].items() if v != ""}
                est = cls(**clean_params)
                if scale:
                    est = Pipeline([("scaler", StandardScaler()), ("model", est)])
                X_full = df[feature_cols].copy()
                X_full = encode_and_fill(X_full)
                X_full = apply_fe(X_full, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs)
                y_full = df[target_col].copy()
                if comp_task == "Classification":
                    y_full = LabelEncoder().fit_transform(y_full.astype(str))
                else:
                    y_full = pd.to_numeric(y_full, errors="coerce")
                est.fit(X_full, y_full)
                out_df = df.copy()
                out_df["prediction"] = est.predict(X_full)
                st.download_button("Download CSV", out_df.to_csv(index=False).encode(),
                                   "predictions.csv","text/csv")
            except Exception as e:
                st.error(f"Could not export: {e}")
