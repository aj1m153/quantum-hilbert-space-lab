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

task = st.selectbox("Task type", [
    "Classification", "Regression", "Clustering",
    "Time Series (statsmodels)", "Prophet"
])

# ── CLASSIFICATION / REGRESSION ───────────────────────────────────────────────
if task in ("Classification", "Regression"):
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  precision_score, recall_score,
                                  mean_squared_error, r2_score, mean_absolute_error,
                                  confusion_matrix)

    CLF_MODELS = {
        "Logistic Regression": ("sklearn.linear_model", "LogisticRegression", {"C": [0.01,0.1,1,10], "max_iter":[500]}),
        "Random Forest": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators":[50,100,200], "max_depth":[None,5,10]}),
        "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingClassifier", {"n_estimators":[50,100], "learning_rate":[0.05,0.1,0.2]}),
        "XGBoost": ("xgboost", "XGBClassifier", {"n_estimators":[50,100], "max_depth":[3,6], "learning_rate":[0.05,0.1]}),
        "LightGBM": ("lightgbm", "LGBMClassifier", {"n_estimators":[50,100], "num_leaves":[31,63]}),
        "SVM": ("sklearn.svm", "SVC", {"C":[0.1,1,10], "kernel":["rbf","linear"]}),
        "KNN": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors":[3,5,7,11]}),
        "Decision Tree": ("sklearn.tree", "DecisionTreeClassifier", {"max_depth":[None,5,10,20]}),
        "Naive Bayes": ("sklearn.naive_bayes", "GaussianNB", {}),
        "Extra Trees": ("sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators":[50,100], "max_depth":[None,5,10]}),
    }
    REG_MODELS = {
        "Linear Regression": ("sklearn.linear_model", "LinearRegression", {}),
        "Ridge": ("sklearn.linear_model", "Ridge", {"alpha":[0.1,1,10,100]}),
        "Lasso": ("sklearn.linear_model", "Lasso", {"alpha":[0.01,0.1,1,10]}),
        "ElasticNet": ("sklearn.linear_model", "ElasticNet", {"alpha":[0.1,1], "l1_ratio":[0.2,0.5,0.8]}),
        "Random Forest": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators":[50,100,200], "max_depth":[None,5,10]}),
        "Gradient Boosting": ("sklearn.ensemble", "GradientBoostingRegressor", {"n_estimators":[50,100], "learning_rate":[0.05,0.1]}),
        "XGBoost": ("xgboost", "XGBRegressor", {"n_estimators":[50,100], "max_depth":[3,6], "learning_rate":[0.05,0.1]}),
        "LightGBM": ("lightgbm", "LGBMRegressor", {"n_estimators":[50,100], "num_leaves":[31,63]}),
        "SVR": ("sklearn.svm", "SVR", {"C":[0.1,1,10], "kernel":["rbf","linear"]}),
        "KNN": ("sklearn.neighbors", "KNeighborsRegressor", {"n_neighbors":[3,5,7,11]}),
    }
    MODEL_CATALOGUE = CLF_MODELS if task == "Classification" else REG_MODELS

    cols = df.columns.tolist()
    target_col = st.selectbox("Target column", cols)
    feature_cols = st.multiselect("Feature columns", [c for c in cols if c != target_col],
                                   default=[c for c in cols if c != target_col])

    if not feature_cols:
        st.info("Select at least one feature column.")
        st.stop()

    model_name = st.selectbox("Model", list(MODEL_CATALOGUE.keys()))
    module_name, class_name, default_grid = MODEL_CATALOGUE[model_name]

    st.markdown("---")
    col_seed, col_split = st.columns(2)
    seed = col_seed.number_input("Random seed", 0, 9999, 42)
    test_size = col_split.slider("Test set size (%)", 10, 40, 20) / 100
    scale = st.checkbox("Scale features (StandardScaler)", value=True)

    st.markdown("---")
    tuning = st.radio("Hyperparameter tuning", ["Manual", "GridSearchCV", "Optuna"], horizontal=True)

    manual_params = {}
    if tuning == "Manual":
        st.markdown("**Set parameters manually:**")
        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            import inspect
            sig = inspect.signature(cls.__init__)
            shown = 0
            cols_p = st.columns(3)
            for pname, param in sig.parameters.items():
                if pname in ("self", "args", "kwargs") or shown >= 9:
                    continue
                default = param.default if param.default != inspect.Parameter.empty else None
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    val = cols_p[shown % 3].number_input(pname, value=float(default) if default is not None else 1.0, key=f"mp_{pname}")
                    manual_params[pname] = val
                elif isinstance(default, str):
                    val = cols_p[shown % 3].text_input(pname, value=default, key=f"mp_{pname}")
                    manual_params[pname] = val
                shown += 1
        except Exception:
            st.info("Could not auto-detect parameters. Using model defaults.")

    if tuning == "GridSearchCV" and default_grid:
        st.markdown("**GridSearch parameter grid (JSON):**")
        import json
        grid_str = st.text_area("Edit grid", value=json.dumps(default_grid, indent=2), height=150)
        try:
            param_grid = json.loads(grid_str)
        except Exception:
            param_grid = default_grid
            st.warning("Invalid JSON detected. Using default grid.")

    if tuning == "Optuna":
        n_trials = st.number_input("Number of Optuna trials", 10, 200, 30)

    if st.button("Train Model", type="primary"):
        with st.spinner("Preparing data..."):
            X = df[feature_cols].copy()
            y = df[target_col].copy()

            for c in X.select_dtypes(include=["object","category"]).columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            X = X.fillna(X.median(numeric_only=True))

            if task == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            else:
                y = pd.to_numeric(y, errors="coerce")
                mask = y.notna()
                X, y = X[mask], y[mask]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(seed)
            )

        with st.spinner("Training model..."):
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)

            if tuning == "Manual":
                est = cls(**{k: v for k, v in manual_params.items() if v != ""})
            elif tuning == "GridSearchCV":
                base = cls(random_state=int(seed)) if "random_state" in cls.__init__.__code__.co_varnames else cls()
                cv = GridSearchCV(base, param_grid, cv=3, n_jobs=-1,
                                   scoring="accuracy" if task=="Classification" else "neg_root_mean_squared_error")
                cv.fit(X_train, y_train)
                st.success(f"Best parameters: {cv.best_params_}")
                est = cv.best_estimator_
            elif tuning == "Optuna":
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    params = {}
                    for pname, vals in default_grid.items():
                        params[pname] = trial.suggest_categorical(pname, vals)
                    try:
                        m = cls(**params)
                        scores = cross_val_score(m, X_train, y_train, cv=3,
                            scoring="accuracy" if task=="Classification" else "neg_root_mean_squared_error")
                        return scores.mean()
                    except Exception:
                        return -999

                study = optuna.create_study(direction="maximize")
                progress = st.progress(0, text="Optuna optimization running...")
                for i in range(int(n_trials)):
                    study.optimize(objective, n_trials=1)
                    progress.progress((i+1)/int(n_trials), text=f"Trial {i+1} of {n_trials}")
                progress.empty()
                best_params = study.best_params
                st.success(f"Best parameters: {best_params}")
                est = cls(**best_params)

            if scale:
                from sklearn.pipeline import Pipeline as Pipe
                est = Pipe([("scaler", StandardScaler()), ("model", est)])

            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)

        # ── Results ───────────────────────────────────────────────────────────
        st.subheader("Results")

        if task == "Classification":
            acc       = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            try:
                if len(np.unique(y_test)) == 2:
                    proba = est.predict_proba(X_test)[:,1] if hasattr(est,"predict_proba") else y_pred
                    auc = roc_auc_score(y_test, proba)
                else:
                    proba = est.predict_proba(X_test) if hasattr(est,"predict_proba") else None
                    auc = roc_auc_score(y_test, proba, multi_class="ovr") if proba is not None else None
            except Exception:
                auc = None

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{acc:.4f}")
            c2.metric("Precision", f"{precision:.4f}")
            c3.metric("Recall",    f"{recall:.4f}")
            c4.metric("F1 Score",  f"{f1:.4f}")
            if auc:
                c5.metric("ROC-AUC", f"{auc:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig, use_container_width=False)
            plt.close()

            metrics_dict = {
                "accuracy":  acc,
                "precision": precision,
                "recall":    recall,
                "f1_weighted": f1,
            }
            if auc:
                metrics_dict["roc_auc"] = auc

        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAE", f"{mae:.4f}")
            c3.metric("R2 Score", f"{r2:.4f}")

            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_test, y_pred, alpha=0.5, s=20)
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([mn,mx],[mn,mx],"r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig, use_container_width=False)
            plt.close()
            metrics_dict = {"rmse": rmse, "mae": mae, "r2": r2}

        # ── Feature importance ────────────────────────────────────────────────
        raw_model = est.named_steps["model"] if scale and hasattr(est, "named_steps") else est
        if hasattr(raw_model, "feature_importances_"):
            fi = pd.Series(raw_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi)*0.3)))
            fi.plot(kind="barh", ax=ax2)
            ax2.set_title("Feature Importances")
            st.pyplot(fig2, use_container_width=False)
            plt.close()
        elif hasattr(raw_model, "coef_"):
            coef = raw_model.coef_.flatten()[:len(feature_cols)]
            fi = pd.Series(coef, index=feature_cols[:len(coef)]).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi)*0.3)))
            fi.plot(kind="barh", ax=ax2)
            ax2.set_title("Model Coefficients")
            st.pyplot(fig2, use_container_width=False)
            plt.close()

        # ── SHAP ──────────────────────────────────────────────────────────────
        try:
            import shap
            shap_model = raw_model
            X_sample = X_test[:100] if len(X_test) > 100 else X_test
            if hasattr(shap_model, "predict_proba") or hasattr(shap_model, "predict"):
                explainer = shap.TreeExplainer(shap_model) if hasattr(shap_model, "feature_importances_") \
                            else shap.LinearExplainer(shap_model, X_train)
                shap_vals = explainer.shap_values(X_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                fig3, ax3 = plt.subplots()
                shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols,
                                  plot_type="bar", show=False)
                st.subheader("SHAP Feature Importance")
                st.pyplot(fig3, use_container_width=False)
                plt.close()
        except Exception:
            pass

        # ── Save to session ───────────────────────────────────────────────────
        result_entry = {
            "name": f"{model_name} ({task})",
            "model": est,
            "task": task,
            "metrics": metrics_dict,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
        if "model_results" not in st.session_state:
            st.session_state["model_results"] = []
        st.session_state["model_results"].append(result_entry)
        st.session_state["last_model"] = result_entry

        buf = io.BytesIO()
        pickle.dump(est, buf)
        st.download_button(
            "Download trained model (.pkl)",
            buf.getvalue(),
            f"{model_name.replace(' ','_')}.pkl"
        )

# ── CLUSTERING ────────────────────────────────────────────────────────────────
elif task == "Clustering":
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    feature_cols = st.multiselect("Feature columns", df.columns.tolist(),
                                   default=df.select_dtypes(include=np.number).columns.tolist())
    algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

    if algo == "KMeans":
        k = st.slider("Number of clusters (k)", 2, 15, 3)
        seed = st.number_input("Random seed", 0, 9999, 42)
    elif algo == "DBSCAN":
        eps = st.number_input("eps", 0.1, 10.0, 0.5)
        min_samples = st.number_input("min_samples", 1, 50, 5)
    else:
        k = st.slider("Number of clusters", 2, 15, 3)
        linkage = st.selectbox("Linkage", ["ward","complete","average","single"])

    scale = st.checkbox("Scale features", True)

    if st.button("Run Clustering", type="primary"):
        X = df[feature_cols].copy()
        for c in X.select_dtypes(include=["object","category"]).columns:
            from sklearn.preprocessing import LabelEncoder
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = X.fillna(X.median(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X) if scale else X.values

        if algo == "KMeans":
            model = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        elif algo == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=int(min_samples))
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        labels = model.fit_predict(X_scaled)
        df["Cluster"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.metric("Clusters Found", n_clusters)
        if n_clusters > 1:
            sil = silhouette_score(X_scaled, labels)
            st.metric("Silhouette Score", f"{sil:.4f}")

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(6,5))
        scatter = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap="tab10", alpha=0.7, s=20)
        ax.set_title("Cluster Visualization (PCA 2D Projection)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig, use_container_width=False)
        plt.close()

        st.session_state["df"] = df
        st.success("Cluster labels added to the dataset as a 'Cluster' column.")

# ── TIME SERIES ───────────────────────────────────────────────────────────────
elif task == "Time Series (statsmodels)":
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    date_col = st.selectbox("Date/time column", df.columns.tolist())
    value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns.tolist())
    model_type = st.selectbox("Model", ["SARIMA", "Exponential Smoothing (Holt-Winters)"])
    forecast_periods = st.slider("Forecast periods", 5, 60, 12)

    if model_type == "SARIMA":
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("p", 0, 5, 1)
        d = c2.number_input("d", 0, 2, 1)
        q = c3.number_input("q", 0, 5, 1)
        c4, c5, c6, c7 = st.columns(4)
        P = c4.number_input("P", 0, 3, 1)
        D = c5.number_input("D", 0, 2, 1)
        Q = c6.number_input("Q", 0, 3, 1)
        s = c7.number_input("s", 1, 52, 12)

    if st.button("Fit and Forecast", type="primary"):
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna().sort_values(date_col).set_index(date_col)
        series = ts_df[value_col]

        with st.spinner("Fitting model..."):
            if model_type == "SARIMA":
                model = SARIMAX(series, order=(int(p),int(d),int(q)),
                                seasonal_order=(int(P),int(D),int(Q),int(s)),
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
                forecast = result.forecast(steps=forecast_periods)
                st.text(result.summary().as_text()[:2000])
            else:
                model = ExponentialSmoothing(series, trend="add", seasonal="add",
                                             seasonal_periods=12)
                result = model.fit()
                forecast = result.forecast(forecast_periods)

        fig, ax = plt.subplots(figsize=(10,4))
        series.plot(ax=ax, label="Historical")
        forecast.plot(ax=ax, label="Forecast", color="red", linestyle="--")
        ax.set_title("Time Series Forecast")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.dataframe(forecast.rename("Forecast").reset_index(), use_container_width=True)

# ── PROPHET ───────────────────────────────────────────────────────────────────
elif task == "Prophet":
    date_col = st.selectbox("Date column (ds)", df.columns.tolist())
    value_col = st.selectbox("Value column (y)", df.select_dtypes(include=np.number).columns.tolist())
    forecast_periods = st.slider("Forecast periods (days)", 30, 730, 90)
    changepoint_scale = st.slider("Changepoint prior scale", 0.01, 0.5, 0.05)
    seasonality_mode = st.selectbox("Seasonality mode", ["additive","multiplicative"])

    if st.button("Fit Prophet Model", type="primary"):
        try:
            from prophet import Prophet
        except ImportError:
            st.error("Prophet is not installed. Run: pip install prophet")
            st.stop()

        prophet_df = df[[date_col, value_col]].rename(
            columns={date_col: "ds", value_col: "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
        prophet_df = prophet_df.dropna()

        with st.spinner("Fitting Prophet model..."):
            m = Prophet(changepoint_prior_scale=changepoint_scale,
                        seasonality_mode=seasonality_mode)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_periods)
            forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.subheader("Forecast Table")
        st.dataframe(
            forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(forecast_periods),
            use_container_width=True
        )
