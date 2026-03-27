import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StrandSense Analytics",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A2D 0%, #2D7A4F 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #b8e0c8; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .kpi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2D7A4F;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .kpi-val   { font-size: 1.8rem; font-weight: 700; color: #1B3A2D; }
    .kpi-label { font-size: 0.8rem; color: #6b7280; margin-top: 0.2rem; }
    .section-header {
        background: #f0faf4;
        border-left: 4px solid #2D7A4F;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
        color: #1B3A2D;
        font-size: 1.05rem;
    }
    .insight-box {
        background: #f0faf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #1B3A2D;
    }
    .warn-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #9a3412;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_GREEN  = "#2D7A4F"
C_DARK   = "#1B3A2D"
C_LIGHT  = "#D4EDE1"
C_COLORS = ["#2D7A4F","#F4A261","#8B5CF6","#06B6D4","#EF4444","#F59E0B"]

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset_haircare.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💇 StrandSense")
    st.markdown("**D2C Haircare Analytics**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 EDA & Descriptive Stats",
         "🤖 Classification Model",
         "🔗 Association Rule Mining",
         "🎯 Clustering & Segmentation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"- Rows: `{len(df)}`")
    st.markdown(f"- Columns: `{df.shape[1]}`")
    st.markdown(f"- Conversions: `{int(df['Converted'].sum())}`")
    st.markdown(f"- Conv Rate: `{df['Converted'].mean():.1%}`")
    st.markdown("---")
    st.caption("StrandSense © 2024 · Data Analytics Assignment")

# ── Helper: KPI card ───────────────────────────────────────────────────────────
def kpi(col, label, val, color=C_GREEN):
    with col:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:{color}">'
            f'<div class="kpi-val" style="color:{color}">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

def sec(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>💇 StrandSense Haircare</h1>
        <p>End-to-End D2C Business Validation Dashboard &nbsp;·&nbsp; Data Analytics Assignment</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(6)
    kpi(cols[0], "Total Leads",   f"{len(df):,}")
    kpi(cols[1], "Conversions",   f"{int(df['Converted'].sum()):,}",  "#8B5CF6")
    kpi(cols[2], "Conv. Rate",    f"{df['Converted'].mean():.1%}",    C_GREEN)
    kpi(cols[3], "Revenue",       f"₹{df['Revenue_Generated'].sum():,.0f}", "#F4A261")
    kpi(cols[4], "Avg Order",     f"₹{df[df['Order_Value_INR'].notna()]['Order_Value_INR'].mean():,.0f}", "#06B6D4")
    kpi(cols[5], "Avg Rating",    f"{df['Customer_Rating'].mean():.2f}/5", "#EF4444")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        sec("📦 Product Portfolio")
        products = {
            "Anti-Hair Fall Shampoo": "₹699",
            "Onion & Biotin Serum":   "₹949",
            "Deep Repair Hair Mask":  "₹849",
            "Scalp Detox Oil":        "₹799",
            "Keratin Conditioner":    "₹749",
        }
        for prod, price in products.items():
            st.markdown(f"- **{prod}** — {price}")

    with right:
        sec("🗂️ Analytics Pipeline")
        for step, desc in [
            ("1. Synthetic Data",    "300 leads · 15 features · haircare domain logic"),
            ("2. Data Cleaning",     "Typos, nulls, outliers & invalid ranges fixed"),
            ("3. Feature Engineering","Month, Quarter, Revenue_Generated derived"),
            ("4. EDA",               "6 chart types · correlation matrix · channel stats"),
            ("5. Classification",    "Random Forest · Accuracy/Precision/Recall/F1/AUC"),
            ("6. Association Rules", "Apriori · Support / Confidence / Lift"),
            ("7. Clustering",        "K-Means · Elbow · PCA · Silhouette score"),
        ]:
            st.markdown(f"**{step}** — {desc}")

    sec("🧭 App Modules")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.info("**📊 EDA**\n\nDistributions, trends, correlations and channel performance")
    with c2: st.info("**🤖 Classification**\n\nConversion prediction with Random Forest + full metrics")
    with c3: st.info("**🔗 Association**\n\nApriori rules — Support, Confidence, Lift analysis")
    with c4: st.info("**🎯 Clustering**\n\nK-Means segments with PCA and silhouette scoring")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Descriptive Stats":
    st.markdown("""
    <div class="main-header">
        <h1>📊 EDA & Descriptive Statistics</h1>
        <p>Exploratory analysis of the StrandSense D2C haircare sales funnel</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    kpi(cols[0], "Total Leads",   f"{len(df):,}")
    kpi(cols[1], "Converted",     f"{int(df['Converted'].sum())}", "#8B5CF6")
    kpi(cols[2], "Total Revenue", f"₹{df['Revenue_Generated'].sum():,.0f}", "#F4A261")
    kpi(cols[3], "Repeat Rate",   f"{df[df['Converted']==1]['Repeat_Purchase'].mean():.1%}", "#06B6D4")

    st.markdown("<br>", unsafe_allow_html=True)

    # Descriptive stats
    sec("📋 Descriptive Statistics — Numeric Variables")
    num_cols = ["Sessions_Before_Purchase","Ad_Spend_INR","Order_Value_INR",
                "Customer_Rating","Revenue_Generated"]
    st.dataframe(
        df[num_cols].describe().round(2).style.background_gradient(cmap="Greens", axis=1),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        sec("📡 Conversion Rate by Acquisition Channel")
        ch = (df.groupby("Acquisition_Channel")["Converted"]
              .mean().reset_index().sort_values("Converted", ascending=True))
        fig = px.bar(ch, x="Converted", y="Acquisition_Channel", orientation="h",
                     color="Converted", color_continuous_scale=["#b7e4c7", C_DARK],
                     text=ch["Converted"].map("{:.1%}".format),
                     labels={"Converted":"Conversion Rate","Acquisition_Channel":"Channel"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=40,t=10,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
        insight("<b>Organic Search (56.7%)</b> and <b>Email (50.0%)</b> lead conversion. "
                "Influencer has the highest ad cost yet the lowest conversion rate — a clear ROI gap.")

    with c2:
        sec("💰 Total Revenue by Product SKU")
        pr = (df.groupby("Product_Interest")["Revenue_Generated"]
              .sum().reset_index().sort_values("Revenue_Generated", ascending=True))
        fig2 = px.bar(pr, x="Revenue_Generated", y="Product_Interest", orientation="h",
                      color="Revenue_Generated", color_continuous_scale=["#d4ede1", C_DARK],
                      text=pr["Revenue_Generated"].map("₹{:,.0f}".format),
                      labels={"Revenue_Generated":"Revenue (₹)","Product_Interest":"Product"})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=70,t=10,b=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)
        insight("<b>Deep Repair Hair Mask</b> and <b>Keratin Conditioner</b> together contribute ~48% of revenue. "
                "Anti-Hair Fall Shampoo (entry SKU) underperforms — bundle strategy needed.")

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        sec("🔍 Lead Distribution by Hair Concern")
        hc = df.groupby("Hair_Concern")["Lead_ID"].count().reset_index()
        fig3 = px.pie(hc, names="Hair_Concern", values="Lead_ID",
                      color_discrete_sequence=C_COLORS, hole=0.4)
        fig3.update_traces(textposition="outside", textinfo="percent+label")
        fig3.update_layout(showlegend=True, margin=dict(l=10,r=10,t=10,b=10), height=350,
                           paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
        insight("<b>Hair Thinning</b> is the most common concern (24.7%), but "
                "<b>Dry & Frizzy</b> converts at 47.3% — highest of any segment.")

    with c4:
        sec("👥 Conversion Rate by Age Group")
        age = (df.groupby("Age_Group")["Converted"]
               .mean().reset_index().sort_values("Age_Group"))
        fig4 = px.bar(age, x="Age_Group", y="Converted",
                      color="Converted", color_continuous_scale=["#b7e4c7", C_DARK],
                      text=age["Converted"].map("{:.1%}".format),
                      labels={"Converted":"Conversion Rate","Age_Group":"Age Group"})
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=350,
                           yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)
        insight("<b>35-44 age group</b> converts at 43% — highest among standard-sized cohorts. "
                "Income + urgency around hair health drives premium purchase intent.")

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        sec("📅 Monthly Revenue Trend (2024)")
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly = df.groupby("Month")["Revenue_Generated"].sum().reset_index()
        monthly["Month_Name"] = monthly["Month"].map(month_map)
        fig5 = px.line(monthly, x="Month_Name", y="Revenue_Generated", markers=True,
                       color_discrete_sequence=[C_GREEN],
                       labels={"Revenue_Generated":"Revenue (₹)","Month_Name":"Month"})
        fig5.update_traces(line_width=3, marker_size=8)
        fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=320)
        st.plotly_chart(fig5, use_container_width=True)
        insight("Revenue peaks in <b>Q2</b> (pre-summer scalp concerns) and <b>Q4</b> "
                "(post-monsoon hair fall recovery). Seasonal campaigns should front-load spend in Mar & Sep.")

    with c6:
        sec("📊 Sessions Before Purchase — Converted vs Not")
        fig6 = px.histogram(df, x="Sessions_Before_Purchase", color="Conversion_Flag",
                            barmode="overlay", nbins=7,
                            color_discrete_map={"Yes": C_GREEN, "No": "#F4A261"},
                            labels={"Sessions_Before_Purchase":"Sessions","Conversion_Flag":"Converted"})
        fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=320)
        st.plotly_chart(fig6, use_container_width=True)
        insight("Both converters and non-converters show similar session distributions — "
                "conversion is driven by <b>channel intent</b> and <b>product fit</b>, not browse depth.")

    # Correlation heatmap
    sec("🔥 Pearson Correlation Heatmap")
    corr_cols = ["Sessions_Before_Purchase","Ad_Spend_INR","Order_Value_INR",
                 "Customer_Rating","Converted","Revenue_Generated","Repeat_Purchase"]
    corr_matrix = df[corr_cols].corr().round(3)
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale=["#EF4444","white", C_GREEN],
                         zmin=-1, zmax=1)
    fig_corr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=420)
    st.plotly_chart(fig_corr, use_container_width=True)

    ci1, ci2 = st.columns(2)
    with ci1:
        insight("<b>Ad Spend ↔ Converted = -0.18:</b> Higher-cost channels convert LESS efficiently. "
                "Reallocate budget from Influencer to Email and SEO.")
    with ci2:
        insight("<b>Rating ↔ Repeat Purchase = +0.28:</b> Satisfied customers repurchase more — "
                "product quality is the best retention lever.")

    # Channel table
    sec("📋 Channel Performance Summary Table")
    ch_tbl = df.groupby("Acquisition_Channel").agg(
        Leads=("Lead_ID","count"),
        Conversions=("Converted","sum"),
        Avg_Ad_Spend=("Ad_Spend_INR","mean"),
        Total_Revenue=("Revenue_Generated","sum"),
    ).reset_index()
    ch_tbl["Conv_Rate"]     = (ch_tbl["Conversions"]/ch_tbl["Leads"]).map("{:.1%}".format)
    ch_tbl["Avg_Ad_Spend"]  = ch_tbl["Avg_Ad_Spend"].map("₹{:,.0f}".format)
    ch_tbl["Total_Revenue"] = ch_tbl["Total_Revenue"].map("₹{:,.0f}".format)
    ch_tbl.columns = ["Channel","Leads","Conversions","Avg Ad Spend","Total Revenue","Conv. Rate"]
    st.dataframe(
        ch_tbl[["Channel","Leads","Conversions","Conv. Rate","Avg Ad Spend","Total Revenue"]],
        use_container_width=True, hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Classification Model":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, roc_curve,
                                  confusion_matrix, classification_report)

    st.markdown("""
    <div class="main-header">
        <h1>🤖 Classification Model</h1>
        <p>Random Forest — Predicting Customer Conversion with full performance metrics</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Model Settings**")
        n_estimators = st.slider("Number of Trees",  50,  300, 100, 50)
        max_depth    = st.slider("Max Depth",          2,   15,   6)
        test_size    = st.slider("Test Set %",        15,   40,  25) / 100

    # Feature preparation
    @st.cache_data
    def build_features(data):
        use_cols = ["Acquisition_Channel","Region","Age_Group","Gender","Device_Type",
                    "Product_Interest","Hair_Concern","Sessions_Before_Purchase",
                    "Ad_Spend_INR","Month","Quarter","Converted"]
        fdf = data[use_cols].dropna().copy()
        cats = ["Acquisition_Channel","Region","Age_Group","Gender",
                "Device_Type","Product_Interest","Hair_Concern"]
        for c in cats:
            fdf[c] = LabelEncoder().fit_transform(fdf[c].astype(str))
        return fdf

    fdf = build_features(df)
    feat_cols = [c for c in fdf.columns if c != "Converted"]
    X = fdf[feat_cols].values
    y = fdf["Converted"].values

    @st.cache_data
    def run_rf(n_est, depth, ts):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                     random_state=42, class_weight="balanced")
        clf.fit(Xtr, ytr)
        return clf, Xtr, Xte, ytr, yte

    clf, X_train, X_test, y_train, y_test = run_rf(n_estimators, max_depth, test_size)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    # Metric cards
    sec("📈 Classification Performance Metrics")
    m_cols = st.columns(5)
    kpi(m_cols[0], "Accuracy",  f"{acc:.3f}",  C_GREEN)
    kpi(m_cols[1], "Precision", f"{prec:.3f}", "#8B5CF6")
    kpi(m_cols[2], "Recall",    f"{rec:.3f}",  "#F4A261")
    kpi(m_cols[3], "F1-Score",  f"{f1:.3f}",   "#06B6D4")
    kpi(m_cols[4], "ROC-AUC",   f"{auc:.3f}",  "#EF4444")

    st.markdown("<br>", unsafe_allow_html=True)

    # ROC + Confusion
    c1, c2 = st.columns(2)
    with c1:
        sec("📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
            line=dict(color=C_GREEN, width=3)))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Random Baseline",
            line=dict(color="#9ca3af", width=2, dash="dash")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.5, y=0.08),
            margin=dict(l=10,r=10,t=10,b=10), height=380)
        fig_roc.update_xaxes(showgrid=True, gridcolor="#f0f0f0", range=[0,1])
        fig_roc.update_yaxes(showgrid=True, gridcolor="#f0f0f0", range=[0,1])
        st.plotly_chart(fig_roc, use_container_width=True)
        insight(f"AUC = <b>{auc:.3f}</b> — the model correctly ranks a converted lead above a "
                f"non-converted one <b>{auc:.0%}</b> of the time. Scores above 0.70 indicate a useful classifier.")

    with c2:
        sec("🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True,
                           x=["Predicted: No","Predicted: Yes"],
                           y=["Actual: No","Actual: Yes"],
                           color_continuous_scale=["#f0faf4", C_DARK],
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=10,r=10,t=10,b=10), height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        insight(f"True Positives: <b>{tp}</b> &nbsp;|&nbsp; True Negatives: <b>{tn}</b> "
                f"&nbsp;|&nbsp; False Positives: <b>{fp}</b> &nbsp;|&nbsp; False Negatives: <b>{fn}</b>")

    # Feature Importance
    sec("🏆 Feature Importance")
    fi_df = pd.DataFrame({
        "Feature":    feat_cols,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#d4ede1", C_DARK],
                    labels={"Importance":"Importance Score","Feature":"Feature"},
                    text=fi_df["Importance"].map("{:.3f}".format))
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(showlegend=False, coloraxis_showscale=False,
                         plot_bgcolor="white", paper_bgcolor="white",
                         margin=dict(l=10,r=60,t=10,b=10), height=420)
    st.plotly_chart(fig_fi, use_container_width=True)

    top3 = fi_df.sort_values("Importance", ascending=False).head(3)["Feature"].tolist()
    insight(f"Top 3 predictors of conversion: <b>{', '.join(top3)}</b>. "
            "These features should drive customer scoring models and retargeting logic.")

    # Classification Report
    sec("📋 Detailed Classification Report")
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Not Converted","Converted"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(
        report_df.style.background_gradient(
            cmap="Greens", subset=["precision","recall","f1-score"]),
        use_container_width=True,
    )

    # Metric explanations
    sec("📖 Metric Definitions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("- **Accuracy** — % of all predictions that are correct\n"
                    "- **Precision** — Of predicted conversions, what % truly converted\n"
                    "- **Recall** — Of all actual conversions, what % were captured")
    with d2:
        st.markdown("- **F1-Score** — Harmonic mean of Precision & Recall\n"
                    "- **ROC-AUC** — Model's ability to separate classes (1.0 = perfect)\n"
                    "- **Feature Importance** — How much each feature reduces tree impurity")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rule Mining":
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    st.markdown("""
    <div class="main-header">
        <h1>🔗 Association Rule Mining</h1>
        <p>Apriori algorithm — discovering co-occurrence patterns across products, channels and hair concerns</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Apriori Settings**")
        min_support    = st.slider("Min Support",    0.05, 0.50, 0.10, 0.01)
        min_confidence = st.slider("Min Confidence", 0.10, 0.90, 0.35, 0.05)
        min_lift       = st.slider("Min Lift",       1.00, 3.00, 1.05, 0.05)

    sec("🧩 What is Association Rule Mining?")
    st.markdown("""
    Association Rule Mining finds **"if A then B"** patterns in transactional data:
    - **Support** = Frequency of the itemset in all transactions (how common)
    - **Confidence** = P(B | A) — if A occurs, probability B also occurs
    - **Lift** = How much more likely than random chance (>1.0 = genuine positive association)
    """)

    @st.cache_data
    def build_transactions():
        conv_df = df[df["Converted"] == 1].copy()
        transactions = []
        for _, row in conv_df.iterrows():
            items = [
                f"PROD:{row['Product_Interest']}",
                f"CH:{row['Acquisition_Channel']}",
                f"CONCERN:{row['Hair_Concern']}",
                f"AGE:{row['Age_Group']}",
                f"REGION:{row['Region']}",
                f"DEVICE:{row['Device_Type']}",
                "REPEAT:Yes" if row["Repeat_Purchase"] == 1 else "REPEAT:No",
                "SESSIONS:High" if row["Sessions_Before_Purchase"] >= 4 else "SESSIONS:Low",
            ]
            transactions.append(items)
        return transactions

    transactions = build_transactions()
    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    te_df  = pd.DataFrame(te_arr, columns=te.columns_)

    try:
        freq_items = apriori(te_df, min_support=min_support, use_colnames=True)

        if len(freq_items) == 0:
            st.warning("No frequent itemsets found. Lower the Min Support slider.")
            st.stop()

        rules = association_rules(
            freq_items, metric="lift", min_threshold=min_lift,
            num_itemsets=len(freq_items))
        rules = rules[rules["confidence"] >= min_confidence].copy()
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        disp = rules[["antecedents","consequents","support","confidence","lift"]].copy()
        disp["support"]    = disp["support"].round(4)
        disp["confidence"] = disp["confidence"].round(4)
        disp["lift"]       = disp["lift"].round(4)

        # Stat cards
        st.markdown("<br>", unsafe_allow_html=True)
        sc = st.columns(4)
        kpi(sc[0], "Frequent Itemsets", str(len(freq_items)))
        kpi(sc[1], "Rules Generated",   str(len(disp)),           "#8B5CF6")
        kpi(sc[2], "Max Lift",
            f"{disp['lift'].max():.2f}" if len(disp) else "—",     "#F4A261")
        kpi(sc[3], "Max Confidence",
            f"{disp['confidence'].max():.2%}" if len(disp) else "—","#06B6D4")

        st.markdown("<br>", unsafe_allow_html=True)

        if len(disp) == 0:
            st.warning("No rules survive current thresholds. Try lowering Min Confidence or Min Lift.")
        else:
            # Rules table
            sec("📋 All Association Rules (sorted by Lift ↓)")
            st.dataframe(
                disp.head(40).style
                    .background_gradient(subset=["lift"],       cmap="Greens")
                    .background_gradient(subset=["confidence"], cmap="Blues")
                    .background_gradient(subset=["support"],    cmap="Oranges"),
                use_container_width=True, hide_index=True,
            )

            # Scatter
            sec("🔵 Support vs Confidence — bubble size = Lift")
            fig_sc = px.scatter(
                disp, x="support", y="confidence",
                size="lift", color="lift",
                color_continuous_scale=["#b7e4c7", C_DARK],
                hover_data=["antecedents","consequents","lift"],
                labels={"support":"Support","confidence":"Confidence","lift":"Lift"},
                size_max=35,
            )
            fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  margin=dict(l=10,r=10,t=10,b=10), height=430)
            st.plotly_chart(fig_sc, use_container_width=True)

            # Top-10 by Lift
            sec("🏆 Top 10 Rules by Lift")
            top10 = disp.head(10).copy()
            top10["Rule"] = (top10["antecedents"].str[:32] + "  →  " +
                             top10["consequents"].str[:28])
            fig_lift = px.bar(
                top10.sort_values("lift"), x="lift", y="Rule", orientation="h",
                color="lift", color_continuous_scale=["#d4ede1", C_GREEN],
                text=top10.sort_values("lift")["lift"].map("{:.2f}".format),
                labels={"lift":"Lift Score","Rule":"Rule"},
            )
            fig_lift.update_traces(textposition="outside")
            fig_lift.update_layout(showlegend=False, coloraxis_showscale=False,
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(l=10,r=60,t=10,b=10), height=430)
            st.plotly_chart(fig_lift, use_container_width=True)

            # Distributions
            dc1, dc2 = st.columns(2)
            with dc1:
                sec("📊 Confidence Distribution")
                fig_conf = px.histogram(disp, x="confidence", nbins=15,
                                        color_discrete_sequence=[C_GREEN],
                                        labels={"confidence":"Confidence"})
                fig_conf.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(l=10,r=10,t=10,b=10), height=300)
                st.plotly_chart(fig_conf, use_container_width=True)

            with dc2:
                sec("📊 Lift Distribution")
                fig_liftd = px.histogram(disp, x="lift", nbins=15,
                                          color_discrete_sequence=["#8B5CF6"],
                                          labels={"lift":"Lift"})
                fig_liftd.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                        margin=dict(l=10,r=10,t=10,b=10), height=300)
                st.plotly_chart(fig_liftd, use_container_width=True)

            # Interpretation guide
            sec("💡 Interpretation Guide")
            ig1, ig2, ig3 = st.columns(3)
            with ig1:
                insight("<b>Support > 0.10</b> — Pattern appears in 10%+ of converted transactions; statistically meaningful frequency.")
            with ig2:
                insight("<b>Confidence > 0.35</b> — Given antecedent, consequent appears 35%+ of the time; strong conditional probability.")
            with ig3:
                insight("<b>Lift > 1.1</b> — Items co-occur 10%+ more than random chance; a genuine positive association.")

    except Exception as e:
        st.error(f"Association mining error: {e}")
        st.info("Try adjusting the sliders in the sidebar.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Clustering & Segmentation":
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    st.markdown("""
    <div class="main-header">
        <h1>🎯 Clustering & Customer Segmentation</h1>
        <p>K-Means clustering with PCA visualisation — identifying distinct customer personas</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Clustering Settings**")
        n_clusters  = st.slider("Number of Clusters (K)", 2, 6, 3)
        include_all = st.checkbox("Include non-converters", value=True)

    @st.cache_data
    def prep_cluster(include_all_flag):
        cdf = df.copy() if include_all_flag else df[df["Converted"] == 1].copy()
        cdf = cdf.reset_index(drop=True)
        num_feats = ["Sessions_Before_Purchase","Ad_Spend_INR",
                     "Converted","Repeat_Purchase","Month","Quarter"]
        cat_feats = ["Acquisition_Channel","Age_Group","Hair_Concern","Device_Type"]
        enc_df = cdf[num_feats + cat_feats].copy()
        for c in cat_feats:
            enc_df[c] = LabelEncoder().fit_transform(enc_df[c].astype(str))
        enc_df = enc_df.fillna(enc_df.median(numeric_only=True))
        return enc_df, cdf

    enc_df, base_df = prep_cluster(include_all)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(enc_df)

    @st.cache_data
    def compute_kmeans(k, flag):
        ed, bd = prep_cluster(flag)
        sc = StandardScaler()
        Xs = sc.fit_transform(ed)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lbl) if k > 1 else 0.0
        pc  = PCA(n_components=2, random_state=42)
        coords = pc.fit_transform(Xs)
        return lbl, sil, coords, km, pc

    labels, sil_score, pca_coords, km_model, pca_model = compute_kmeans(n_clusters, include_all)
    base_df = base_df.reset_index(drop=True)
    base_df["Cluster"]       = labels
    base_df["Cluster_Label"] = (base_df["Cluster"] + 1).apply(lambda x: f"Cluster {x}")
    base_df["PCA_1"]         = pca_coords[:, 0]
    base_df["PCA_2"]         = pca_coords[:, 1]

    # Metric cards
    sc_cols = st.columns(4)
    kpi(sc_cols[0], "Clusters (K)",       str(n_clusters))
    kpi(sc_cols[1], "Silhouette Score",   f"{sil_score:.3f}",  "#8B5CF6")
    kpi(sc_cols[2], "Total Customers",    f"{len(base_df):,}", "#F4A261")
    kpi(sc_cols[3], "PCA Variance Expl.", f"{sum(pca_model.explained_variance_ratio_):.1%}", "#06B6D4")

    st.markdown("<br>", unsafe_allow_html=True)

    # Elbow + PCA
    c1, c2 = st.columns(2)
    with c1:
        sec("📐 Elbow Method & Silhouette Score")

        @st.cache_data
        def elbow(flag):
            ed, _ = prep_cluster(flag)
            sc = StandardScaler()
            Xs = sc.fit_transform(ed)
            ks, inertias, sils = [], [], []
            for k in range(2, 8):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl = km.fit_predict(Xs)
                ks.append(k)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(Xs, lbl))
            return ks, inertias, sils

        ks, inertias, sils = elbow(include_all)
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia (WCSS)",
                       line=dict(color=C_GREEN, width=3), marker=dict(size=8)),
            secondary_y=False)
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=sils, mode="lines+markers", name="Silhouette Score",
                       line=dict(color="#F4A261", width=3, dash="dash"), marker=dict(size=8)),
            secondary_y=True)
        fig_elbow.add_vline(x=n_clusters, line_dash="dot", line_color="#8B5CF6",
                            annotation_text=f"K={n_clusters} selected")
        fig_elbow.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig_elbow.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                legend=dict(x=0.55, y=0.9),
                                margin=dict(l=10,r=10,t=10,b=10), height=360)
        st.plotly_chart(fig_elbow, use_container_width=True)
        insight("Elbow = where inertia drops steeply then flattens. "
                "Pick K where <b>silhouette score peaks</b> — higher = better-defined clusters.")

    with c2:
        sec("🔵 PCA 2D Cluster Visualisation")
        fig_pca = px.scatter(
            base_df, x="PCA_1", y="PCA_2", color="Cluster_Label",
            color_discrete_sequence=C_COLORS,
            hover_data=["Acquisition_Channel","Hair_Concern","Age_Group"],
            labels={"PCA_1":"PCA Component 1","PCA_2":"PCA Component 2"},
        )
        fig_pca.update_traces(marker=dict(size=7, opacity=0.75))
        fig_pca.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=10,r=10,t=10,b=10), height=360)
        st.plotly_chart(fig_pca, use_container_width=True)
        insight(f"PCA reduces all features to 2 dimensions. "
                f"Variance explained: <b>{sum(pca_model.explained_variance_ratio_):.1%}</b>. "
                "Well-separated blobs = distinct customer personas.")

    # Cluster profiles
    sec("📊 Cluster Profile Summary")
    profile = base_df.groupby("Cluster_Label").agg(
        Size=("Cluster_Label","count"),
        Conv_Rate=("Converted","mean"),
        Avg_Sessions=("Sessions_Before_Purchase","mean"),
        Avg_AdSpend=("Ad_Spend_INR","mean"),
        Repeat_Rate=("Repeat_Purchase","mean"),
    ).reset_index()
    profile["Conv_Rate"]    = profile["Conv_Rate"].map("{:.1%}".format)
    profile["Repeat_Rate"]  = profile["Repeat_Rate"].map("{:.1%}".format)
    profile["Avg_Sessions"] = profile["Avg_Sessions"].round(1)
    profile["Avg_AdSpend"]  = profile["Avg_AdSpend"].map("₹{:,.0f}".format)
    profile.columns = ["Cluster","Size","Conv. Rate","Avg Sessions","Avg Ad Spend","Repeat Rate"]
    st.dataframe(profile, use_container_width=True, hide_index=True)

    # Cluster size + Conversion rate bars
    cc1, cc2 = st.columns(2)
    with cc1:
        sec("📊 Cluster Size")
        sz = base_df.groupby("Cluster_Label").size().reset_index(name="Count")
        fig_sz = px.bar(sz, x="Cluster_Label", y="Count", color="Cluster_Label",
                        color_discrete_sequence=C_COLORS, text="Count",
                        labels={"Cluster_Label":"Cluster","Count":"Customers"})
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=10,r=10,t=10,b=10), height=320)
        st.plotly_chart(fig_sz, use_container_width=True)

    with cc2:
        sec("📊 Conversion Rate by Cluster")
        cr = base_df.groupby("Cluster_Label")["Converted"].mean().reset_index()
        fig_cr = px.bar(cr, x="Cluster_Label", y="Converted", color="Cluster_Label",
                        color_discrete_sequence=C_COLORS,
                        text=cr["Converted"].map("{:.1%}".format),
                        labels={"Cluster_Label":"Cluster","Converted":"Conv. Rate"})
        fig_cr.update_traces(textposition="outside")
        fig_cr.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=10,r=10,t=10,b=10), height=320,
                             yaxis_tickformat=".0%")
        st.plotly_chart(fig_cr, use_container_width=True)

    # Heatmap: Cluster × Hair Concern
    sec("🔥 Cluster × Hair Concern Heatmap")
    hc_heat = (base_df.groupby(["Cluster_Label","Hair_Concern"])
               .size().reset_index(name="Count"))
    hc_pivot = hc_heat.pivot(index="Cluster_Label", columns="Hair_Concern",
                              values="Count").fillna(0)
    fig_heat = px.imshow(hc_pivot, text_auto=True,
                         color_continuous_scale=["#f0faf4", C_DARK],
                         labels=dict(x="Hair Concern", y="Cluster", color="Count"))
    fig_heat.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(fig_heat, use_container_width=True)
    insight("Darker cells = high concentration of that hair concern in the cluster. "
            "Use this to craft <b>cluster-specific product recommendations</b> and messaging campaigns.")

    # Channel mix by cluster
    sec("📡 Acquisition Channel Mix by Cluster")
    ch_mix = (base_df.groupby(["Cluster_Label","Acquisition_Channel"])
              .size().reset_index(name="Count"))
    fig_ch = px.bar(ch_mix, x="Cluster_Label", y="Count", color="Acquisition_Channel",
                    barmode="group", color_discrete_sequence=C_COLORS,
                    labels={"Cluster_Label":"Cluster","Count":"Leads",
                            "Acquisition_Channel":"Channel"})
    fig_ch.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                         margin=dict(l=10,r=10,t=10,b=10), height=360)
    st.plotly_chart(fig_ch, use_container_width=True)
    insight("Clusters dominated by <b>Organic Search</b> are high-intent and need minimal persuasion spend. "
            "Clusters skewed toward <b>Influencer/Instagram</b> can be retargeted via cost-efficient Email flows.")
