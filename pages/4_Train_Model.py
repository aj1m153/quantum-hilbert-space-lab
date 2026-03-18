import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, pickle, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Train Model", layout="wide")
st.title("Train Model")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def encode_and_fill(X):
    from sklearn.preprocessing import LabelEncoder
    for c in X.select_dtypes(include=["object","category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    return X

def apply_feature_engineering(X, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs):
    import itertools
    X = X.copy()
    if "Log transform" in fe_options:
        for c in log_cols:
            if c in X.columns:
                X[f"log_{c}"] = np.log1p(X[c].clip(lower=0))
    if "Binning (cut into groups)" in fe_options:
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
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        poly_arr = poly.fit_transform(X[num_cols])
        poly_names = poly.get_feature_names_out(num_cols)
        poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=X.index)
        X = pd.concat([X.drop(columns=num_cols), poly_df], axis=1)
    return X

# ═════════════════════════════════════════════════════════════════════════════
# TASK SELECTOR
# ═════════════════════════════════════════════════════════════════════════════
task = st.selectbox("Select task type", [
    "Classification", "Regression", "Clustering",
    "Time Series", "Prophet"
])

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols     = df.columns.tolist()

# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
if task == "Classification":
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  precision_score, recall_score,
                                  confusion_matrix, classification_report,
                                  roc_curve, precision_recall_curve)

    CLF_MODELS = {
        "Logistic Regression":  ("sklearn.linear_model",  "LogisticRegression",         {"C":[0.01,0.1,1,10],           "max_iter":[500]}),
        "Random Forest":        ("sklearn.ensemble",       "RandomForestClassifier",     {"n_estimators":[50,100,200],   "max_depth":[None,5,10]}),
        "Gradient Boosting":    ("sklearn.ensemble",       "GradientBoostingClassifier", {"n_estimators":[50,100],       "learning_rate":[0.05,0.1,0.2]}),
        "XGBoost":              ("xgboost",                "XGBClassifier",              {"n_estimators":[50,100],       "max_depth":[3,6], "learning_rate":[0.05,0.1]}),
        "LightGBM":             ("lightgbm",               "LGBMClassifier",             {"n_estimators":[50,100],       "num_leaves":[31,63]}),
        "Decision Tree":        ("sklearn.tree",           "DecisionTreeClassifier",     {"max_depth":[None,5,10,20]}),
        "Extra Trees":          ("sklearn.ensemble",       "ExtraTreesClassifier",       {"n_estimators":[50,100],       "max_depth":[None,5,10]}),
        "SVM":                  ("sklearn.svm",            "SVC",                        {"C":[0.1,1,10],                "kernel":["rbf","linear"]}),
        "KNN":                  ("sklearn.neighbors",      "KNeighborsClassifier",       {"n_neighbors":[3,5,7,11]}),
        "Naive Bayes":          ("sklearn.naive_bayes",    "GaussianNB",                 {}),
    }

    with st.expander("1. Data Configuration", expanded=True):
        target_col   = st.selectbox("Target column", all_cols, key="clf_target")
        feature_cols = st.multiselect("Feature columns", [c for c in all_cols if c != target_col],
                                       default=[c for c in all_cols if c != target_col], key="clf_features")
        c1, c2, c3 = st.columns(3)
        seed      = c1.number_input("Random seed", 0, 9999, 42, key="clf_seed")
        test_size = c2.slider("Test set size (%)", 10, 40, 20, key="clf_split") / 100
        scale     = c3.checkbox("Scale features (StandardScaler)", value=True, key="clf_scale")

    with st.expander("2. Feature Engineering", expanded=False):
        fe_options = st.multiselect("Select transformations", 
            ["Log transform","Binning (cut into groups)","Interaction terms","Polynomial features"],
            key="clf_fe")
        log_cols, bin_cols, bin_bins, poly_degree, interact_pairs = [], [], 4, 2, []
        if "Log transform" in fe_options:
            log_cols = st.multiselect("Columns to log-transform", numeric_cols, key="clf_log")
        if "Binning (cut into groups)" in fe_options:
            bin_cols = st.multiselect("Columns to bin", numeric_cols, key="clf_bin")
            bin_bins = st.slider("Number of bins", 2, 20, 4, key="clf_bins")
        if "Interaction terms" in fe_options:
            st.markdown("Select pairs of columns to multiply together:")
            n_pairs = st.number_input("Number of interaction pairs", 1, 10, 1, key="clf_npairs")
            for i in range(int(n_pairs)):
                cc1, cc2 = st.columns(2)
                a = cc1.selectbox(f"Pair {i+1} — column A", numeric_cols, key=f"clf_ia_{i}")
                b = cc2.selectbox(f"Pair {i+1} — column B", numeric_cols, key=f"clf_ib_{i}")
                interact_pairs.append((a, b))
        if "Polynomial features" in fe_options:
            poly_degree = st.slider("Polynomial degree", 2, 4, 2, key="clf_poly")

    with st.expander("3. Model Selection", expanded=True):
        model_name = st.selectbox("Model", list(CLF_MODELS.keys()), key="clf_model")
        module_name, class_name, default_grid = CLF_MODELS[model_name]

    with st.expander("4. Hyperparameter Tuning", expanded=False):
        tuning = st.radio("Tuning method", ["Manual", "GridSearchCV", "Optuna"], horizontal=True, key="clf_tuning")
        manual_params = {}
        if tuning == "Manual":
            st.markdown("Set parameters manually:")
            try:
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name)
                import inspect
                sig    = inspect.signature(cls.__init__)
                shown  = 0
                pcols  = st.columns(3)
                for pname, param in sig.parameters.items():
                    if pname in ("self","args","kwargs") or shown >= 9:
                        continue
                    default = param.default if param.default != inspect.Parameter.empty else None
                    if isinstance(default, (int, float)) and not isinstance(default, bool):
                        manual_params[pname] = pcols[shown%3].number_input(pname, value=float(default) if default else 1.0, key=f"clf_mp_{pname}")
                    elif isinstance(default, str):
                        manual_params[pname] = pcols[shown%3].text_input(pname, value=default, key=f"clf_mp_{pname}")
                    shown += 1
            except Exception:
                st.info("Using model defaults.")
        elif tuning == "GridSearchCV":
            import json
            grid_str   = st.text_area("Parameter grid (JSON)", value=json.dumps(default_grid, indent=2), height=150, key="clf_grid")
            try:    param_grid = json.loads(grid_str)
            except: param_grid = default_grid
        elif tuning == "Optuna":
            n_trials = st.number_input("Number of trials", 10, 200, 30, key="clf_trials")

    if st.button("Train Classification Model", type="primary", key="clf_train"):
        with st.spinner("Preparing data..."):
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            X = encode_and_fill(X)
            X = apply_feature_engineering(X, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs)
            X = X.fillna(0)
            le = LabelEncoder()
            y  = le.fit_transform(y.astype(str))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed))

        with st.spinner("Training..."):
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            if tuning == "Manual":
                est = cls(**{k: v for k, v in manual_params.items() if v != ""})
            elif tuning == "GridSearchCV":
                base = cls(random_state=int(seed)) if "random_state" in cls.__init__.__code__.co_varnames else cls()
                cv   = GridSearchCV(base, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
                cv.fit(X_train, y_train)
                st.success(f"Best parameters: {cv.best_params_}")
                est = cv.best_estimator_
            elif tuning == "Optuna":
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                def objective(trial):
                    params = {p: trial.suggest_categorical(p, v) for p, v in default_grid.items()}
                    try:
                        m = cls(**params)
                        return cross_val_score(m, X_train, y_train, cv=3, scoring="accuracy").mean()
                    except: return -999
                study = optuna.create_study(direction="maximize")
                prog  = st.progress(0)
                for i in range(int(n_trials)):
                    study.optimize(objective, n_trials=1)
                    prog.progress((i+1)/int(n_trials))
                prog.empty()
                st.success(f"Best parameters: {study.best_params}")
                est = cls(**study.best_params)

            if scale:
                est = Pipeline([("scaler", StandardScaler()), ("model", est)])
            est.fit(X_train, y_train)
            y_pred  = est.predict(X_test)
            is_binary = len(np.unique(y_test)) == 2

            try:
                y_prob = est.predict_proba(X_test)[:,1] if is_binary else est.predict_proba(X_test)
                has_prob = True
            except: has_prob = False; y_prob = None

        # ── Core metrics ──────────────────────────────────────────────────────
        st.subheader("Results")
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc  = None
        if has_prob:
            try:
                auc = roc_auc_score(y_test, y_prob if is_binary else y_prob, multi_class="ovr" if not is_binary else "raise")
            except: pass

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy",  f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall",    f"{rec:.4f}")
        c4.metric("F1 Score",  f"{f1:.4f}")
        if auc: c5.metric("ROC-AUC", f"{auc:.4f}")

        # ── KS Statistic & Somers D ───────────────────────────────────────────
        if has_prob and is_binary:
            from scipy import stats as scipy_stats
            ks_stat, ks_pval = scipy_stats.ks_2samp(
                y_prob[y_test == 1], y_prob[y_test == 0]
            )
            somers_d = 2 * auc - 1 if auc else None
            k1, k2 = st.columns(2)
            k1.metric("KS Statistic", f"{ks_stat:.4f}", help=f"p-value: {ks_pval:.4f}")
            if somers_d: k2.metric("Somers D", f"{somers_d:.4f}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        st.markdown("**Confusion Matrix**")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i,j], ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax); plt.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close()

        # ── Classification report ─────────────────────────────────────────────
        st.markdown("**Classification Report**")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        if has_prob and is_binary:
            col_left, col_right = st.columns(2)

            # ── ROC-AUC curve ─────────────────────────────────────────────────
            with col_left:
                st.markdown("**ROC-AUC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}" if auc else "ROC")
                ax.plot([0,1],[0,1],"k--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            # ── Precision-Recall curve ────────────────────────────────────────
            with col_right:
                st.markdown("**Precision-Recall Curve**")
                prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(rec_c, prec_c, lw=2)
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve"); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            col_left2, col_right2 = st.columns(2)

            # ── Lift curve ────────────────────────────────────────────────────
            with col_left2:
                st.markdown("**Lift Curve**")
                sorted_idx    = np.argsort(y_prob)[::-1]
                sorted_labels = y_test[sorted_idx]
                pct_pop  = np.arange(1, len(sorted_labels)+1) / len(sorted_labels)
                pct_pos  = np.cumsum(sorted_labels) / sorted_labels.sum()
                baseline = pct_pop
                lift     = pct_pos / baseline
                fig, ax  = plt.subplots(figsize=(5, 4))
                ax.plot(pct_pop * 100, lift, lw=2)
                ax.axhline(1, color="red", linestyle="--", lw=1, label="Baseline")
                ax.set_xlabel("% Population"); ax.set_ylabel("Lift")
                ax.set_title("Lift Curve"); ax.legend(); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            # ── Cumulative gains curve ────────────────────────────────────────
            with col_right2:
                st.markdown("**Cumulative Gains Curve**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(pct_pop * 100, pct_pos * 100, lw=2, label="Model")
                ax.plot([0,100],[0,100],"k--", lw=1, label="Baseline")
                ax.set_xlabel("% Population"); ax.set_ylabel("% Positive Cases Captured")
                ax.set_title("Cumulative Gains"); ax.legend(); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            # ── KS Plot ───────────────────────────────────────────────────────
            st.markdown("**KS Plot**")
            thresholds  = np.linspace(0, 1, 100)
            tpr_list, fpr_list = [], []
            for t in thresholds:
                pred_t = (y_prob >= t).astype(int)
                tp = ((pred_t==1)&(y_test==1)).sum()
                fp = ((pred_t==1)&(y_test==0)).sum()
                fn = ((pred_t==0)&(y_test==1)).sum()
                tn = ((pred_t==0)&(y_test==0)).sum()
                tpr_list.append(tp/(tp+fn) if (tp+fn) > 0 else 0)
                fpr_list.append(fp/(fp+tn) if (fp+tn) > 0 else 0)
            ks_diff = np.abs(np.array(tpr_list) - np.array(fpr_list))
            ks_max_idx = np.argmax(ks_diff)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(thresholds, tpr_list, label="TPR (Sensitivity)")
            ax.plot(thresholds, fpr_list, label="FPR (1-Specificity)")
            ax.axvline(thresholds[ks_max_idx], color="red", linestyle="--",
                       label=f"KS = {ks_diff[ks_max_idx]:.4f} at {thresholds[ks_max_idx]:.2f}")
            ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
            ax.set_title("KS Plot"); ax.legend(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        # ── Feature importance / SHAP ─────────────────────────────────────────
        raw = est.named_steps["model"] if scale and hasattr(est,"named_steps") else est
        feat_names = list(X_train.columns)
        if hasattr(raw,"feature_importances_"):
            fi = pd.Series(raw.feature_importances_, index=feat_names).nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(6, max(3, len(fi)*0.3)))
            fi.plot(kind="barh", ax=ax); ax.set_title("Feature Importances"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()
        elif hasattr(raw,"coef_"):
            coef = pd.Series(np.abs(raw.coef_.flatten()[:len(feat_names)]), index=feat_names[:len(raw.coef_.flatten())]).nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(6, max(3, len(coef)*0.3)))
            coef.plot(kind="barh", ax=ax); ax.set_title("Coefficients (abs)"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()

        try:
            import shap
            X_s = X_test[:100]
            explainer = shap.TreeExplainer(raw) if hasattr(raw,"feature_importances_") else shap.LinearExplainer(raw, X_train)
            sv = explainer.shap_values(X_s)
            if isinstance(sv, list): sv = sv[1]
            fig3, _ = plt.subplots()
            shap.summary_plot(sv, X_s, feature_names=feat_names, plot_type="bar", show=False)
            st.markdown("**SHAP Feature Importance**")
            st.pyplot(fig3, use_container_width=False); plt.close()
        except: pass

        # ── Save & download ───────────────────────────────────────────────────
        result_entry = {"name": f"{model_name} (Classification)", "model": est, "task": "Classification",
                        "metrics": {"accuracy":acc,"precision":prec,"recall":rec,"f1_weighted":f1},
                        "feature_cols": list(X_train.columns), "target_col": target_col}
        if auc: result_entry["metrics"]["roc_auc"] = auc
        st.session_state.setdefault("model_results",[]).append(result_entry)
        buf = io.BytesIO(); pickle.dump(est, buf)
        st.download_button("Download trained model (.pkl)", buf.getvalue(), f"{model_name.replace(' ','_')}.pkl")

# ═════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Regression":
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    REG_MODELS = {
        "Linear Regression":  ("sklearn.linear_model","LinearRegression",          {}),
        "Ridge":              ("sklearn.linear_model","Ridge",                     {"alpha":[0.1,1,10,100]}),
        "Lasso":              ("sklearn.linear_model","Lasso",                     {"alpha":[0.01,0.1,1,10]}),
        "ElasticNet":         ("sklearn.linear_model","ElasticNet",                {"alpha":[0.1,1],"l1_ratio":[0.2,0.5,0.8]}),
        "Random Forest":      ("sklearn.ensemble",    "RandomForestRegressor",     {"n_estimators":[50,100,200],"max_depth":[None,5,10]}),
        "Gradient Boosting":  ("sklearn.ensemble",    "GradientBoostingRegressor", {"n_estimators":[50,100],"learning_rate":[0.05,0.1]}),
        "XGBoost":            ("xgboost",             "XGBRegressor",              {"n_estimators":[50,100],"max_depth":[3,6],"learning_rate":[0.05,0.1]}),
        "LightGBM":           ("lightgbm",            "LGBMRegressor",             {"n_estimators":[50,100],"num_leaves":[31,63]}),
        "SVR":                ("sklearn.svm",         "SVR",                       {"C":[0.1,1,10],"kernel":["rbf","linear"]}),
        "KNN":                ("sklearn.neighbors",   "KNeighborsRegressor",       {"n_neighbors":[3,5,7,11]}),
    }

    with st.expander("1. Data Configuration", expanded=True):
        target_col   = st.selectbox("Target column", numeric_cols, key="reg_target")
        feature_cols = st.multiselect("Feature columns", [c for c in all_cols if c != target_col],
                                       default=[c for c in all_cols if c != target_col], key="reg_features")
        c1, c2, c3 = st.columns(3)
        seed      = c1.number_input("Random seed", 0, 9999, 42, key="reg_seed")
        test_size = c2.slider("Test set size (%)", 10, 40, 20, key="reg_split") / 100
        scale     = c3.checkbox("Scale features (StandardScaler)", value=True, key="reg_scale")

    with st.expander("2. Feature Engineering", expanded=False):
        fe_options = st.multiselect("Select transformations",
            ["Log transform","Binning (cut into groups)","Interaction terms","Polynomial features"], key="reg_fe")
        log_cols, bin_cols, bin_bins, poly_degree, interact_pairs = [], [], 4, 2, []
        if "Log transform" in fe_options:
            log_cols = st.multiselect("Columns to log-transform", numeric_cols, key="reg_log")
        if "Binning (cut into groups)" in fe_options:
            bin_cols  = st.multiselect("Columns to bin", numeric_cols, key="reg_bin")
            bin_bins  = st.slider("Number of bins", 2, 20, 4, key="reg_bins")
        if "Interaction terms" in fe_options:
            n_pairs = st.number_input("Number of interaction pairs", 1, 10, 1, key="reg_npairs")
            for i in range(int(n_pairs)):
                cc1, cc2 = st.columns(2)
                a = cc1.selectbox(f"Pair {i+1} — column A", numeric_cols, key=f"reg_ia_{i}")
                b = cc2.selectbox(f"Pair {i+1} — column B", numeric_cols, key=f"reg_ib_{i}")
                interact_pairs.append((a,b))
        if "Polynomial features" in fe_options:
            poly_degree = st.slider("Polynomial degree", 2, 4, 2, key="reg_poly")

    with st.expander("3. Model Selection", expanded=True):
        model_name = st.selectbox("Model", list(REG_MODELS.keys()), key="reg_model")
        module_name, class_name, default_grid = REG_MODELS[model_name]

    with st.expander("4. Hyperparameter Tuning", expanded=False):
        tuning = st.radio("Tuning method", ["Manual","GridSearchCV","Optuna"], horizontal=True, key="reg_tuning")
        manual_params = {}
        if tuning == "Manual":
            try:
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name)
                import inspect
                sig = inspect.signature(cls.__init__)
                shown = 0; pcols = st.columns(3)
                for pname, param in sig.parameters.items():
                    if pname in ("self","args","kwargs") or shown >= 9: continue
                    default = param.default if param.default != inspect.Parameter.empty else None
                    if isinstance(default,(int,float)) and not isinstance(default,bool):
                        manual_params[pname] = pcols[shown%3].number_input(pname, value=float(default) if default else 1.0, key=f"reg_mp_{pname}")
                    elif isinstance(default,str):
                        manual_params[pname] = pcols[shown%3].text_input(pname, value=default, key=f"reg_mp_{pname}")
                    shown += 1
            except: st.info("Using model defaults.")
        elif tuning == "GridSearchCV":
            import json
            grid_str = st.text_area("Parameter grid (JSON)", value=json.dumps(default_grid,indent=2), height=150, key="reg_grid")
            try:    param_grid = json.loads(grid_str)
            except: param_grid = default_grid
        elif tuning == "Optuna":
            n_trials = st.number_input("Number of trials", 10, 200, 30, key="reg_trials")

    with st.expander("5. Regression Diagnostics to Run", expanded=False):
        run_vif      = st.checkbox("VIF (multicollinearity)", value=True)
        run_dw       = st.checkbox("Durbin-Watson (autocorrelation)", value=True)
        run_bp       = st.checkbox("Breusch-Pagan (heteroscedasticity)", value=True)
        run_normtest = st.checkbox("Normality test (Shapiro-Wilk on residuals)", value=True)
        run_qq       = st.checkbox("QQ plot", value=True)

    if st.button("Train Regression Model", type="primary", key="reg_train"):
        with st.spinner("Preparing data..."):
            X = df[feature_cols].copy()
            y = pd.to_numeric(df[target_col], errors="coerce")
            X = encode_and_fill(X)
            X = apply_feature_engineering(X, fe_options, log_cols, bin_cols, bin_bins, poly_degree, interact_pairs)
            X = X.fillna(0)
            mask = y.notna(); X, y = X[mask], y[mask]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed))

        with st.spinner("Training..."):
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            if tuning == "Manual":
                est = cls(**{k:v for k,v in manual_params.items() if v != ""})
            elif tuning == "GridSearchCV":
                base = cls(random_state=int(seed)) if "random_state" in cls.__init__.__code__.co_varnames else cls()
                cv   = GridSearchCV(base, param_grid, cv=3, n_jobs=-1, scoring="neg_root_mean_squared_error")
                cv.fit(X_train, y_train)
                st.success(f"Best parameters: {cv.best_params_}")
                est = cv.best_estimator_
            elif tuning == "Optuna":
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                def objective(trial):
                    params = {p: trial.suggest_categorical(p, v) for p, v in default_grid.items()}
                    try:
                        m = cls(**params)
                        return cross_val_score(m, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error").mean()
                    except: return -9999
                study = optuna.create_study(direction="maximize")
                prog  = st.progress(0)
                for i in range(int(n_trials)):
                    study.optimize(objective, n_trials=1)
                    prog.progress((i+1)/int(n_trials))
                prog.empty()
                st.success(f"Best parameters: {study.best_params}")
                est = cls(**study.best_params)

            if scale:
                est = Pipeline([("scaler",StandardScaler()),("model",est)])
            est.fit(X_train, y_train)
            y_pred   = est.predict(X_test)
            residuals = y_test.values - y_pred

        # ── Metrics ───────────────────────────────────────────────────────────
        st.subheader("Results")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test.values - y_pred) / np.where(y_test.values==0,1,y_test.values))) * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("RMSE", f"{rmse:.4f}")
        c2.metric("MAE",  f"{mae:.4f}")
        c3.metric("R2",   f"{r2:.4f}")
        c4.metric("MAPE", f"{mape:.2f}%")

        col1, col2 = st.columns(2)
        # ── Actual vs Predicted ───────────────────────────────────────────────
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_test, y_pred, alpha=0.5, s=20)
            mn, mx = min(y_test.min(),y_pred.min()), max(y_test.max(),y_pred.max())
            ax.plot([mn,mx],[mn,mx],"r--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted"); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        # ── Residuals vs Fitted ───────────────────────────────────────────────
        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_pred, residuals, alpha=0.5, s=20)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted"); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        # ── QQ Plot ───────────────────────────────────────────────────────────
        if run_qq:
            from scipy import stats as sp
            fig, ax = plt.subplots(figsize=(5,4))
            sp.probplot(residuals, dist="norm", plot=ax)
            ax.set_title("QQ Plot of Residuals"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()

        # ── Residual histogram ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(residuals, bins=40, edgecolor="none", alpha=0.8)
        ax.set_xlabel("Residual"); ax.set_title("Residual Distribution"); plt.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close()

        # ── Statistical tests ─────────────────────────────────────────────────
        st.markdown("**Regression Diagnostic Tests**")
        from scipy import stats as sp

        diag_results = {}
        if run_normtest:
            if len(residuals) <= 5000:
                stat, pval = sp.shapiro(residuals)
                diag_results["Shapiro-Wilk (normality)"] = f"Statistic={stat:.4f}, p={pval:.4f} — {'Normal' if pval>0.05 else 'Not normal'}"
            else:
                stat, pval = sp.normaltest(residuals)
                diag_results["D'Agostino (normality)"] = f"Statistic={stat:.4f}, p={pval:.4f} — {'Normal' if pval>0.05 else 'Not normal'}"

        if run_dw:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(residuals)
            diag_results["Durbin-Watson (autocorrelation)"] = f"{dw:.4f} — {'No autocorrelation' if 1.5<dw<2.5 else 'Possible autocorrelation'}"

        if run_bp:
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                X_bp = X_test.copy().fillna(0)
                bp_lm, bp_pval, _, _ = het_breuschpagan(residuals, X_bp)
                diag_results["Breusch-Pagan (heteroscedasticity)"] = f"LM={bp_lm:.4f}, p={bp_pval:.4f} — {'Homoscedastic' if bp_pval>0.05 else 'Heteroscedastic'}"
            except Exception as e:
                diag_results["Breusch-Pagan"] = f"Could not compute: {e}"

        if run_vif:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X_vif = X_train.copy().fillna(0)
                X_vif = X_vif.select_dtypes(include=np.number)
                vif_data = pd.DataFrame({
                    "Feature": X_vif.columns,
                    "VIF":     [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
                }).sort_values("VIF", ascending=False)
                st.markdown("**VIF (Variance Inflation Factor)**")
                st.dataframe(vif_data.style.highlight_max(subset=["VIF"], color="#f8d7da"), use_container_width=True)
            except Exception as e:
                st.warning(f"VIF could not be computed: {e}")

        for test_name, result in diag_results.items():
            st.info(f"**{test_name}**: {result}")

        # ── Feature importance ────────────────────────────────────────────────
        raw = est.named_steps["model"] if scale and hasattr(est,"named_steps") else est
        feat_names = list(X_train.columns)
        if hasattr(raw,"feature_importances_"):
            fi = pd.Series(raw.feature_importances_, index=feat_names).nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(6, max(3,len(fi)*0.3)))
            fi.plot(kind="barh", ax=ax); ax.set_title("Feature Importances"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()
        elif hasattr(raw,"coef_"):
            coef = pd.Series(raw.coef_.flatten()[:len(feat_names)], index=feat_names[:len(raw.coef_.flatten())]).sort_values()
            fig, ax = plt.subplots(figsize=(6, max(3,len(coef)*0.3)))
            coef.plot(kind="barh", ax=ax); ax.set_title("Coefficients"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()

        # ── Save & download ───────────────────────────────────────────────────
        result_entry = {"name": f"{model_name} (Regression)", "model": est, "task": "Regression",
                        "metrics": {"rmse":rmse,"mae":mae,"r2":r2},
                        "feature_cols": list(X_train.columns), "target_col": target_col}
        st.session_state.setdefault("model_results",[]).append(result_entry)
        buf = io.BytesIO(); pickle.dump(est, buf)
        st.download_button("Download trained model (.pkl)", buf.getvalue(), f"{model_name.replace(' ','_')}.pkl")

# ═════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Clustering":
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    with st.expander("1. Data Configuration", expanded=True):
        feature_cols = st.multiselect("Feature columns", all_cols, default=numeric_cols, key="clu_features")
        scale        = st.checkbox("Scale features", True, key="clu_scale")

    with st.expander("2. Algorithm & Parameters", expanded=True):
        algo = st.selectbox("Algorithm", ["KMeans","DBSCAN","Agglomerative"], key="clu_algo")
        if algo == "KMeans":
            k    = st.slider("Number of clusters (k)", 2, 15, 3, key="clu_k")
            seed = st.number_input("Random seed", 0, 9999, 42, key="clu_seed")
        elif algo == "DBSCAN":
            eps         = st.number_input("eps", 0.1, 10.0, 0.5, key="clu_eps")
            min_samples = st.number_input("min_samples", 1, 50, 5, key="clu_min")
        else:
            k       = st.slider("Number of clusters", 2, 15, 3, key="clu_k2")
            linkage = st.selectbox("Linkage", ["ward","complete","average","single"], key="clu_link")

    if st.button("Run Clustering", type="primary", key="clu_train"):
        X = df[feature_cols].copy()
        for c in X.select_dtypes(include=["object","category"]).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = X.fillna(X.median(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X) if scale else X.values

        # ── Elbow method (KMeans only) ────────────────────────────────────────
        if algo == "KMeans":
            st.subheader("Elbow Method")
            inertias = []
            k_range  = range(2, min(16, len(X)//2))
            for ki in k_range:
                inertias.append(KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_scaled).inertia_)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(list(k_range), inertias, marker="o")
            ax.set_xlabel("k"); ax.set_ylabel("Inertia")
            ax.set_title("Elbow Plot"); plt.tight_layout()
            st.pyplot(fig, use_container_width=False); plt.close()

        # ── Fit model ─────────────────────────────────────────────────────────
        if algo == "KMeans":
            model = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        elif algo == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=int(min_samples))
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        labels     = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        c1, c2 = st.columns(2)
        c1.metric("Clusters found", n_clusters)
        if n_clusters > 1:
            sil = silhouette_score(X_scaled, labels)
            c2.metric("Silhouette Score", f"{sil:.4f}")

        # ── PCA 2D plot ───────────────────────────────────────────────────────
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(6,5))
        scatter = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap="tab10", alpha=0.7, s=20)
        ax.set_title("Clusters — PCA 2D Projection")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster"); plt.tight_layout()
        st.pyplot(fig, use_container_width=False); plt.close()

        # ── Cluster profile table ─────────────────────────────────────────────
        st.markdown("**Cluster Profile Table**")
        df_clustered             = df[feature_cols].copy()
        df_clustered["Cluster"]  = labels
        profile = df_clustered.groupby("Cluster")[
            [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        ].mean().round(3)
        st.dataframe(profile, use_container_width=True)

        df["Cluster"] = labels
        st.session_state["df"] = df
        st.success("Cluster labels added to dataset as 'Cluster' column.")

# ═════════════════════════════════════════════════════════════════════════════
# TIME SERIES
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Time Series":
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with st.expander("1. Data Configuration", expanded=True):
        date_col         = st.selectbox("Date/time column", all_cols, key="ts_date")
        value_col        = st.selectbox("Value column", numeric_cols, key="ts_val")
        forecast_periods = st.slider("Forecast periods", 5, 60, 12, key="ts_fperiods")
        test_periods     = st.slider("Test periods (held out for accuracy)", 3, 30, 6, key="ts_test")

    with st.expander("2. Model Selection", expanded=True):
        model_type = st.selectbox("Model", ["SARIMA","Exponential Smoothing (Holt-Winters)"], key="ts_model")

        if model_type == "SARIMA":
            order_mode = st.radio("Order selection", ["Manual","Automatic (AIC grid search)"], horizontal=True, key="ts_order")
            if order_mode == "Manual":
                c1,c2,c3 = st.columns(3)
                p = c1.number_input("p",0,5,1,key="ts_p"); d = c2.number_input("d",0,2,1,key="ts_d"); q = c3.number_input("q",0,5,1,key="ts_q")
                c4,c5,c6,c7 = st.columns(4)
                P = c4.number_input("P",0,3,1,key="ts_P"); D = c5.number_input("D",0,2,1,key="ts_D")
                Q = c6.number_input("Q",0,3,1,key="ts_Q"); s = c7.number_input("s",1,52,12,key="ts_s")

    if st.button("Fit and Forecast", type="primary", key="ts_train"):
        ts_df         = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df         = ts_df.dropna().sort_values(date_col).set_index(date_col)
        series        = ts_df[value_col]
        train_series  = series.iloc[:-test_periods]
        test_series   = series.iloc[-test_periods:]

        with st.spinner("Fitting model..."):
            if model_type == "SARIMA":
                if order_mode == "Automatic (AIC grid search)":
                    best_aic, best_order, best_sorder = np.inf, (1,1,1), (0,0,0,0)
                    prog = st.progress(0, text="Searching ARIMA orders...")
                    combos = [(p,d,q) for p in range(3) for d in range(2) for q in range(3)]
                    for i,(pi,di,qi) in enumerate(combos):
                        try:
                            res = SARIMAX(train_series, order=(pi,di,qi), enforce_stationarity=False,
                                          enforce_invertibility=False).fit(disp=False)
                            if res.aic < best_aic:
                                best_aic, best_order = res.aic, (pi,di,qi)
                        except: pass
                        prog.progress((i+1)/len(combos))
                    prog.empty()
                    p,d,q = best_order
                    P,D,Q,s = 0,0,0,0
                    st.success(f"Best ARIMA order: {best_order} (AIC={best_aic:.2f})")

                model  = SARIMAX(train_series, order=(int(p),int(d),int(q)),
                                  seasonal_order=(int(P),int(D),int(Q),int(s)),
                                  enforce_stationarity=False, enforce_invertibility=False)
                result   = model.fit(disp=False)
                forecast = result.forecast(steps=forecast_periods)
                test_pred= result.forecast(steps=test_periods)
                st.text(result.summary().as_text()[:2000])
            else:
                model    = ExponentialSmoothing(train_series, trend="add", seasonal="add", seasonal_periods=12)
                result   = model.fit()
                forecast = result.forecast(forecast_periods)
                test_pred= result.forecast(test_periods)

        # ── Test accuracy ─────────────────────────────────────────────────────
        rmse = np.sqrt(np.mean((test_series.values - test_pred.values[:len(test_series)])**2))
        mae  = np.mean(np.abs(test_series.values - test_pred.values[:len(test_series)]))
        mape = np.mean(np.abs((test_series.values - test_pred.values[:len(test_series)]) /
               np.where(test_series.values==0,1,test_series.values))) * 100
        c1,c2,c3 = st.columns(3)
        c1.metric("Test RMSE", f"{rmse:.4f}")
        c2.metric("Test MAE",  f"{mae:.4f}")
        c3.metric("Test MAPE", f"{mape:.2f}%")

        # ── Forecast plot ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10,4))
        train_series.plot(ax=ax, label="Train")
        test_series.plot(ax=ax, label="Actual (test)", color="orange")
        test_pred[:len(test_series)].plot(ax=ax, label="Predicted (test)", color="green", linestyle="--")
        forecast.plot(ax=ax, label="Forecast", color="red", linestyle="--")
        ax.set_title("Time Series Forecast"); ax.legend(); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── Residuals ─────────────────────────────────────────────────────────
        if model_type == "SARIMA":
            resid = result.resid
            fig, axes = plt.subplots(1,2, figsize=(10,3))
            axes[0].plot(resid); axes[0].set_title("Residuals over time")
            axes[1].hist(resid, bins=30, edgecolor="none"); axes[1].set_title("Residual distribution")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.subheader("Forecast Table")
        st.dataframe(forecast.rename("Forecast").reset_index(), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PROPHET
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Prophet":
    with st.expander("1. Data Configuration", expanded=True):
        date_col         = st.selectbox("Date column (ds)", all_cols, key="pr_date")
        value_col        = st.selectbox("Value column (y)", numeric_cols, key="pr_val")
        forecast_periods = st.slider("Forecast periods (days)", 30, 730, 90, key="pr_fperiods")
        test_periods     = st.slider("Test periods held out", 10, 90, 30, key="pr_test")

    with st.expander("2. Prophet Parameters", expanded=True):
        changepoint_scale  = st.slider("Changepoint prior scale", 0.01, 0.5, 0.05, key="pr_cp")
        seasonality_mode   = st.selectbox("Seasonality mode", ["additive","multiplicative"], key="pr_smode")
        yearly_seasonality = st.checkbox("Yearly seasonality", True, key="pr_yearly")
        weekly_seasonality = st.checkbox("Weekly seasonality", True, key="pr_weekly")
        daily_seasonality  = st.checkbox("Daily seasonality", False, key="pr_daily")

    if st.button("Fit Prophet Model", type="primary", key="pr_train"):
        try:
            from prophet import Prophet
        except ImportError:
            st.error("Prophet is not installed. Add prophet to requirements.txt and redeploy.")
            st.stop()

        prophet_df = df[[date_col, value_col]].rename(columns={date_col:"ds", value_col:"y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
        prophet_df = prophet_df.dropna().sort_values("ds")

        train_df = prophet_df.iloc[:-test_periods]
        test_df  = prophet_df.iloc[-test_periods:]

        with st.spinner("Fitting Prophet..."):
            m = Prophet(
                changepoint_prior_scale=changepoint_scale,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
            )
            m.fit(train_df)
            future   = m.make_future_dataframe(periods=forecast_periods + test_periods)
            forecast = m.predict(future)

        # ── Test accuracy ─────────────────────────────────────────────────────
        test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
        if len(test_forecast) > 0:
            merged  = test_df.merge(test_forecast[["ds","yhat"]], on="ds", how="inner")
            rmse    = np.sqrt(np.mean((merged["y"] - merged["yhat"])**2))
            mae     = np.mean(np.abs(merged["y"] - merged["yhat"]))
            mape    = np.mean(np.abs((merged["y"] - merged["yhat"]) /
                      np.where(merged["y"]==0,1,merged["y"]))) * 100
            c1,c2,c3 = st.columns(3)
            c1.metric("Test RMSE", f"{rmse:.4f}")
            c2.metric("Test MAE",  f"{mae:.4f}")
            c3.metric("Test MAPE", f"{mape:.2f}%")

        # ── Forecast plot ─────────────────────────────────────────────────────
        fig1 = m.plot(forecast)
        plt.title("Prophet Forecast")
        st.pyplot(fig1, use_container_width=True); plt.close()

        # ── Components ────────────────────────────────────────────────────────
        st.subheader("Trend and Seasonality Decomposition")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2, use_container_width=True); plt.close()

        # ── Changepoints ──────────────────────────────────────────────────────
        st.subheader("Changepoint Analysis")
        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(prophet_df["ds"], prophet_df["y"], "k.", alpha=0.3, ms=3, label="Data")
        for cp in m.changepoints:
            ax3.axvline(cp, color="red", alpha=0.4, linewidth=0.8)
        ax3.set_title("Detected Changepoints (red lines)")
        ax3.legend(); plt.tight_layout()
        st.pyplot(fig3, use_container_width=True); plt.close()

        # ── Forecast table ────────────────────────────────────────────────────
        st.subheader("Forecast Table")
        st.dataframe(
            forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(forecast_periods),
            use_container_width=True
        )
