import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "CRIM":    "Per-capita crime rate by town",
    "ZN":      "Proportion of residential land zoned for lots > 25,000 sq.ft",
    "INDUS":   "Proportion of non-retail business acres per town",
    "CHAS":    "Charles River dummy (1 if tract bounds river, 0 otherwise)",
    "NOX":     "Nitric oxide concentration (parts per 10 million)",
    "RM":      "Average number of rooms per dwelling",
    "AGE":     "Proportion of owner-occupied units built before 1940",
    "DIS":     "Weighted distances to Boston employment centres",
    "RAD":     "Index of accessibility to radial highways",
    "TAX":     "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil–teacher ratio by town",
    "B":       "1000(Bk - 0.63)² where Bk = proportion of Black residents",
    "LSTAT":   "% lower-status population",
}
TARGET = "MEDV"   # Median value of owner-occupied homes in $1000s

MODELS = {
    "Linear Regression":       LinearRegression(),
    "Ridge Regression":        Ridge(alpha=1.0),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
}


@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset – from upload or fallback to local file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("HousingData.csv")
        except FileNotFoundError:
            st.error("⚠️  HousingData.csv not found. Please upload the file using the sidebar.")
            st.stop()
    return df


@st.cache_resource
def train_model(model_name: str, X_train, y_train):
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    return model


def preprocess(df: pd.DataFrame):
    """Impute missing values & scale features."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, y, imputer, scaler, X.columns.tolist()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Settings")
    uploaded = st.file_uploader("Upload HousingData.csv", type=["csv"])
    st.markdown("---")
    model_name = st.selectbox("🤖 Model", list(MODELS.keys()), index=2)
    test_size  = st.slider("Test split (%)", 10, 40, 20, step=5) / 100
    st.markdown("---")
    st.caption("Boston Housing Dataset · 506 samples · 13 features")

# ─────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────
df_raw = load_data(uploaded)
X_scaled, y, imputer, scaler, feature_names = preprocess(df_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

model = train_model(model_name, X_train, y_train)
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(MODELS[model_name], X_scaled, y, cv=5,
                         scoring="r2").mean()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("🏠 Boston House Price Prediction")
st.markdown(
    "A mini machine-learning project using the classic **Boston Housing Dataset** "
    "to predict median home prices from neighbourhood features."
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Explorer", "📈 Model Performance", "🔍 Feature Insights", "🏡 Predict Price"]
)

# ══════════════════════════════════════════════
# TAB 1 – DATA EXPLORER
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Raw Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_raw.shape[0])
    col2.metric("Features", df_raw.shape[1] - 1)
    col3.metric("Missing cells", int(df_raw.isnull().sum().sum()))
    col4.metric("Avg price ($k)", f"{df_raw[TARGET].mean():.1f}")

    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df_raw.describe().T.style.background_gradient(cmap="Blues"),
                 use_container_width=True)

    st.subheader("Missing Values")
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x=missing.index, y=missing.values, ax=ax, palette="Reds_r")
        ax.set_ylabel("Missing count")
        ax.set_title("Columns with missing values")
        st.pyplot(fig)
        plt.close()

    st.subheader("Target Distribution (MEDV – Median Home Value $k)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df_raw[TARGET], bins=30, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("MEDV ($k)")
    sns.boxplot(y=df_raw[TARGET], ax=axes[1], color="#4C72B0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
# TAB 2 – MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.subheader(f"Results — {model_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",  f"{r2:.3f}")
    m2.metric("CV R² (5-fold)", f"{cv_r2:.3f}")
    m3.metric("RMSE ($k)", f"{rmse:.2f}")
    m4.metric("MAE ($k)",  f"{mae:.2f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Actual vs Predicted**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.5, color="#4C72B0", edgecolors="w", linewidth=0.3)
        lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual MEDV ($k)")
        ax.set_ylabel("Predicted MEDV ($k)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Residuals Distribution**")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="#DD8452")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Residual ($k)")
        st.pyplot(fig)
        plt.close()

    st.subheader("Model Comparison (CV R²)")
    comp_results = {}
    for nm, mdl in MODELS.items():
        scores = cross_val_score(mdl, X_scaled, y, cv=5, scoring="r2")
        comp_results[nm] = scores.mean()

    comp_df = (
        pd.DataFrame.from_dict(comp_results, orient="index", columns=["CV R²"])
        .sort_values("CV R²", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(comp_df.index, comp_df["CV R²"], color=sns.color_palette("viridis", len(comp_df)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("CV R²")
    ax.set_title("5-Fold Cross-Validation R² by Model")
    for bar, val in zip(bars, comp_df["CV R²"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center")
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
# TAB 3 – FEATURE INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 6))
        corr = df_raw.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Feature Importance (Random Forest)")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        imp_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = sns.color_palette("Blues_r", len(imp_df))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors)
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importances")
        st.pyplot(fig)
        plt.close()

    st.subheader("Feature vs Target (MEDV) Scatter")
    sel_feat = st.selectbox("Select a feature", feature_names, index=feature_names.index("RM"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df_raw[sel_feat], df_raw[TARGET], alpha=0.4, color="#2ca02c", edgecolors="w", linewidth=0.2)
    ax.set_xlabel(f"{sel_feat} – {FEATURE_DESCRIPTIONS.get(sel_feat, '')}")
    ax.set_ylabel("MEDV ($k)")
    ax.set_title(f"{sel_feat} vs MEDV")
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Descriptions")
    desc_df = pd.DataFrame.from_dict(FEATURE_DESCRIPTIONS, orient="index", columns=["Description"])
    desc_df.index.name = "Feature"
    st.dataframe(desc_df, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – INTERACTIVE PREDICTION
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🏡 Predict House Price")
    st.markdown(
        "Adjust the sliders to set neighbourhood features, then click **Predict**."
    )

    df_clean = df_raw.fillna(df_raw.median(numeric_only=True))

    col1, col2, col3 = st.columns(3)
    inputs = {}

    slider_specs = {
        "CRIM":    (float(df_clean["CRIM"].min()),   float(df_clean["CRIM"].max()),   float(df_clean["CRIM"].median())),
        "ZN":      (float(df_clean["ZN"].min()),     float(df_clean["ZN"].max()),     float(df_clean["ZN"].median())),
        "INDUS":   (float(df_clean["INDUS"].min()),  float(df_clean["INDUS"].max()),  float(df_clean["INDUS"].median())),
        "CHAS":    (0.0, 1.0, 0.0),
        "NOX":     (float(df_clean["NOX"].min()),    float(df_clean["NOX"].max()),    float(df_clean["NOX"].median())),
        "RM":      (float(df_clean["RM"].min()),     float(df_clean["RM"].max()),     float(df_clean["RM"].median())),
        "AGE":     (float(df_clean["AGE"].min()),    float(df_clean["AGE"].max()),    float(df_clean["AGE"].median())),
        "DIS":     (float(df_clean["DIS"].min()),    float(df_clean["DIS"].max()),    float(df_clean["DIS"].median())),
        "RAD":     (int(df_clean["RAD"].min()),      int(df_clean["RAD"].max()),      int(df_clean["RAD"].median())),
        "TAX":     (int(df_clean["TAX"].min()),      int(df_clean["TAX"].max()),      int(df_clean["TAX"].median())),
        "PTRATIO": (float(df_clean["PTRATIO"].min()),float(df_clean["PTRATIO"].max()),float(df_clean["PTRATIO"].median())),
        "B":       (float(df_clean["B"].min()),      float(df_clean["B"].max()),      float(df_clean["B"].median())),
        "LSTAT":   (float(df_clean["LSTAT"].min()),  float(df_clean["LSTAT"].max()),  float(df_clean["LSTAT"].median())),
    }

    feature_list = list(slider_specs.keys())
    cols = [col1, col2, col3]
    for i, feat in enumerate(feature_list):
        mn, mx, mid = slider_specs[feat]
        with cols[i % 3]:
            step = 1 if feat in ("RAD", "TAX", "CHAS") else round((mx - mn) / 100, 4)
            inputs[feat] = st.slider(
                f"{feat}",
                min_value=float(mn), max_value=float(mx), value=float(mid), step=float(step),
                help=FEATURE_DESCRIPTIONS.get(feat, ""),
            )

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        input_array = np.array([[inputs[f] for f in feature_names]])
        input_imputed = imputer.transform(input_array)
        input_scaled  = scaler.transform(input_imputed)
        prediction    = model.predict(input_scaled)[0]

        st.markdown("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("🏠 Predicted Median Value", f"${prediction*1000:,.0f}")
        res2.metric("Model used", model_name)
        res3.metric("Training R²", f"{r2:.3f}")

        price_pct = (prediction - df_raw[TARGET].min()) / (df_raw[TARGET].max() - df_raw[TARGET].min())
        st.progress(min(float(price_pct), 1.0), text=f"Price percentile in dataset")

        top_feats = imp_df.nlargest(3, "Importance")["Feature"].tolist()
        st.info(
            f"💡 The three most influential features for this model are: "
            f"**{top_feats[0]}**, **{top_feats[1]}**, and **{top_feats[2]}**."
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Boston House Price Prediction · Mini ML Project · "
    "Built with Streamlit, scikit-learn, seaborn & pandas"
)