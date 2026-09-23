import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Logistics Analytics Dashboard", layout="wide")
st.title("📦 E-Commerce Last-Mile Logistics Analytics")
st.markdown("Predicting and visualizing Service Level Agreement (SLA) breaches in supply chain operations.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.markdown("Adjust the synthetic data volume below.")
selected_volume = st.sidebar.selectbox(
    "Select Dataset Size (Rows):",
    options=[1000, 5000, 10000, 50000, 100000, 200000, 500000],
    index=2  # Defaults to 10,000 for smoother editing performance
)

# ==========================================
# PHASE 1: Dynamic Data Generation
# ==========================================
@st.cache_data
def load_data(num_records):
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    date_offsets = np.random.randint(0, 365, num_records)
    purchase_dates = [start_date + timedelta(days=int(offset)) for offset in date_offsets]
    
    carriers = np.random.choice(['FedEx', 'UPS', 'USPS', 'DHL'], num_records, p=[0.4, 0.3, 0.2, 0.1])
    distances_km = np.random.randint(10, 3000, num_records)
    weights_kg = np.round(np.random.uniform(0.5, 30.0, num_records), 2)
    
    expected_days = np.where(distances_km < 500, 2, np.where(distances_km < 1500, 4, 7))
    delay_factors = (distances_km / 1000) + (weights_kg / 10) + np.random.normal(0, 1.5, num_records)
    actual_days = expected_days + np.round(delay_factors).astype(int)
    actual_days = np.maximum(actual_days, 1)
    
    estimated_delivery_dates = [d + timedelta(days=int(e)) for d, e in zip(purchase_dates, expected_days)]
    actual_delivery_dates = [d + timedelta(days=int(a)) for d, a in zip(purchase_dates, actual_days)]
    
    df = pd.DataFrame({
        'Order_ID': range(100000, 100000 + num_records),
        'Purchase_Date': purchase_dates,
        'Estimated_Delivery': estimated_delivery_dates,
        'Actual_Delivery': actual_delivery_dates,
        'Carrier': carriers,
        'Distance_km': distances_km,
        'Weight_kg': weights_kg
    })
    
    df['Promised_Transit_Days'] = (df['Estimated_Delivery'] - df['Purchase_Date']).dt.days
    df['Actual_Transit_Days'] = (df['Actual_Delivery'] - df['Purchase_Date']).dt.days
    df['SLA_Breach'] = (df['Actual_Delivery'] > df['Estimated_Delivery']).astype(int)
    df['Purchase_Month'] = df['Purchase_Date'].dt.month
    df['Purchase_DayOfWeek'] = df['Purchase_Date'].dt.dayofweek
    df['Is_Weekend_Purchase'] = df['Purchase_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

with st.spinner(f"Generating {selected_volume:,} logistics records..."):
    df_raw = load_data(selected_volume)

# Interactive Data Editor
st.subheader("1. Live Data Editor")
st.markdown("Modify any cell below. The entire dashboard (metrics, charts, and ML model) will instantly recalculate based on your edits. *(Note: Editing very large datasets may take a second to process).*")

# The data_editor returns a brand new dataframe containing any changes made by the user
edited_df = st.data_editor(df_raw, use_container_width=True, num_rows="dynamic")

# Display KPI Metrics based on the EDITED dataframe
st.subheader("Operational Baselines")
col1, col2, col3 = st.columns(3)
breach_rate = edited_df['SLA_Breach'].mean() * 100
col1.metric("Total Orders Processed", f"{len(edited_df):,}")
col2.metric("Current SLA Breach Rate", f"{breach_rate:.2f}%", delta="Live Calculation", delta_color="off")
col3.metric("Avg Transit Delay", f"{(edited_df['Actual_Transit_Days'] - edited_df['Promised_Transit_Days']).mean():.1f} Days")

st.divider()

# ==========================================
# PHASE 2: Exploratory Data Analysis (EDA)
# ==========================================
st.header("2. Identifying Bottlenecks (EDA)")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Breach Rate by Carrier")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # Use edited_df for all visualizations
    carrier_breach = edited_df.groupby('Carrier')['SLA_Breach'].mean().reset_index()
    carrier_breach['SLA_Breach_Pct'] = carrier_breach['SLA_Breach'] * 100
    sns.barplot(x='Carrier', y='SLA_Breach_Pct', data=carrier_breach, palette='Blues_d', ax=ax1)
    ax1.set_ylabel('% Late')
    st.pyplot(fig1)

with col_chart2:
    st.subheader("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    numeric_cols = ['SLA_Breach', 'Distance_km', 'Weight_kg', 'Promised_Transit_Days']
    # Use edited_df for correlations
    sns.heatmap(edited_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

st.divider()

# ==========================================
# PHASE 3: Machine Learning Model
# ==========================================
st.header("3. Predictive Modeling (Random Forest)")
st.markdown("Training a model to predict SLA breaches using the dynamically edited dataset.")

# Remove @st.cache_resource here so the model forces a retrain whenever the data is edited
def train_model_live(input_df):
    df_ml = pd.get_dummies(input_df, columns=['Carrier'], drop_first=True)
    feature_cols = ['Distance_km', 'Weight_kg', 'Purchase_Month', 'Is_Weekend_Purchase'] + \
                   [col for col in df_ml.columns if 'Carrier_' in col]
    X = df_ml[feature_cols]
    y = df_ml['SLA_Breach']
    
    # Check if there is enough variance to train a model (in case user deleted too much)
    if len(y.unique()) < 2:
        return 0, pd.DataFrame()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    return acc, feat_df

with st.spinner("Retraining Random Forest model on your live data..."):
    accuracy, feature_importance_df = train_model_live(edited_df)

if accuracy > 0:
    st.success(f"Model successfully updated with **{accuracy * 100:.2f}% accuracy**.")

    st.subheader("What drives delivery delays?")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax3)
    ax3.set_xlabel("Impact on SLA Breach")
    st.pyplot(fig3)
else:
    st.warning("Not enough variation in the 'SLA_Breach' column to train the model. Ensure there are both 0s and 1s.")