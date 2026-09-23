import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION & SIMPLE STYLING
# ==========================================
st.set_page_config(page_title="Logistics Analytics", layout="wide")

# Inject simple CSS to change the background directly in the code (no extra files needed)
st.markdown("""
<style>
    /* Change main app background to a professional dark navy */
    .stApp {
        background-color: #101820; 
    }
    /* Change sidebar background to a slightly darker shade */
    [data-testid="stSidebar"] {
        background-color: #0b1015;
    }
    /* Customize the top header padding for a cleaner look */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Set chart styles to match the dark aesthetic
sns.set_style("darkgrid", {"axes.facecolor": "#1a242f", "figure.facecolor": "#101820"})
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

st.title("📦 Last-Mile Logistics Analytics")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    selected_volume = st.selectbox(
        "Dataset Volume (Rows)",
        options=[1000, 5000, 10000, 50000, 100000],
        index=2
    )

# ==========================================
# DATA GENERATION
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

df_raw = load_data(selected_volume)

# ==========================================
# NAVIGATION TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📋 Data Manager", "📊 Analytics Dashboard", "🤖 Predictive ML"])

# --- TAB 1: DATA MANAGER ---
with tab1:
    st.data_editor(df_raw, use_container_width=True, num_rows="dynamic", height=500)

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
    col1, col2, col3 = st.columns(3)
    breach_rate = df_raw['SLA_Breach'].mean() * 100
    avg_delay = (df_raw['Actual_Transit_Days'] - df_raw['Promised_Transit_Days']).mean()
    
    col1.metric("Total Orders", f"{len(df_raw):,}")
    col2.metric("SLA Breach Rate", f"{breach_rate:.1f}%")
    col3.metric("Avg Transit Delay", f"{avg_delay:.1f} Days")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Breach Rate by Carrier**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        carrier_breach = df_raw.groupby('Carrier')['SLA_Breach'].mean().reset_index()
        sns.barplot(x='Carrier', y='SLA_Breach', data=carrier_breach, ax=ax1, color="#3498db")
        ax1.set_ylabel("Breach Probability")
        ax1.set_xlabel("")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig1)
        
    with c2:
        st.write("**Feature Correlation**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        numeric_cols = ['SLA_Breach', 'Distance_km', 'Weight_kg', 'Promised_Transit_Days']
        sns.heatmap(df_raw[numeric_cols].corr(), annot=True, cmap="Blues", cbar=False, ax=ax2)
        st.pyplot(fig2)

# --- TAB 3: PREDICTIVE ML ---
with tab3:
    def train_model(input_df):
        df_ml = pd.get_dummies(input_df, columns=['Carrier'], drop_first=True)
        feature_cols = ['Distance_km', 'Weight_kg', 'Purchase_Month', 'Is_Weekend_Purchase'] + \
                       [col for col in df_ml.columns if 'Carrier_' in col]
        X = df_ml[feature_cols]
        y = df_ml['SLA_Breach']
        
        if len(y.unique()) < 2:
            return 0, pd.DataFrame()
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        importances = pd.DataFrame({
            'Feature': X.columns, 
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return acc, importances

    accuracy, feat_df = train_model(df_raw)
    
    if accuracy > 0:
        st.success(f"Model Accuracy: **{accuracy * 100:.1f}%**")
        st.write("**Feature Impact on Delivery Delays**")
        
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax3, color="#2ecc71")
        ax3.set_xlabel("Relative Importance")
        ax3.set_ylabel("")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig3)
