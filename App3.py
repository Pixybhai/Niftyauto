# 🚀 PRO NIFTY PREDICTOR (Global Signals + Smart Features)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import datetime

# ================= CONFIG =================
TICKER = "^NSEI"
SP500 = "^GSPC"
VIX = "^VIX"
CRUDE = "CL=F"
START_DATE = "2015-01-01"
MODEL_FILE = "pro_nifty_model.pkl"
# ==========================================

st.set_page_config(page_title="Pro Nifty Predictor", layout="wide")
st.title("🚀 Pro Nifty Predictor (Global + Smart Features)")


# ================= DATA =================
@st.cache_data(ttl=3600)
def fetch_all_data():
    nifty = yf.download(TICKER, start=START_DATE)
    sp500 = yf.download(SP500, start=START_DATE)
    vix = yf.download(VIX, start=START_DATE)
    crude = yf.download(CRUDE, start=START_DATE)

    # Fix MultiIndex
    for df in [nifty, sp500, vix, crude]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = pd.DataFrame()
    df['nifty'] = nifty['Close']
    df['sp500'] = sp500['Close']
    df['vix'] = vix['Close']
    df['crude'] = crude['Close']

    df = df.dropna()
    return df


# ================= FEATURES =================
def add_features(df):
    data = df.copy()

    # Returns
    data['ret_1'] = data['nifty'].pct_change()
    data['ret_5'] = data['nifty'].pct_change(5)

    # Lag features
    for i in range(1, 11):
        data[f'lag_{i}'] = data['nifty'].shift(i)

    # Technicals
    data['sma_5'] = data['nifty'].rolling(5).mean()
    data['sma_20'] = data['nifty'].rolling(20).mean()
    data['volatility'] = data['nifty'].rolling(5).std()

    # Global signals
    data['sp500_ret'] = data['sp500'].pct_change()
    data['vix_change'] = data['vix'].pct_change()
    data['crude_ret'] = data['crude'].pct_change()

    # Target
    data['target'] = data['nifty'].shift(-1)

    data.dropna(inplace=True)

    X = data.drop('target', axis=1)
    y = data['target']

    return X, y, data


# ================= MODEL =================
def get_model(df):
    try:
        model = load(MODEL_FILE)
        st.success("Loaded saved model")
        return model
    except:
        X, y, _ = add_features(df)
        split = int(len(X) * 0.8)

        X_train, y_train = X[:split], y[:split]

        model = RandomForestRegressor(n_estimators=300, max_depth=18)
        model.fit(X_train, y_train)

        dump(model, MODEL_FILE)
        st.success("Model trained")
        return model


# ================= PREDICTION =================
def predict(model, df):
    X, y, data = add_features(df)

    latest = X.iloc[-1:]
    pred = model.predict(latest)[0]

    last_close = df['nifty'].iloc[-1]

    return last_close, pred


# ================= RUN =================
df = fetch_all_data()
model = get_model(df)
last_close, pred = predict(model, df)

# ================= UI =================
col1, col2 = st.columns(2)
col1.metric("Last Close", f"₹{last_close:.2f}")
col2.metric("Predicted Next Close", f"₹{pred:.2f}", f"{pred-last_close:.2f}")

# ================= CHART =================
st.subheader("📊 Last 30 Days + Prediction")
recent = df['nifty'].tail(30)
next_date = recent.index[-1] + pd.Timedelta(days=1)

fig_df = recent.copy()
fig_df.loc[next_date] = pred

st.line_chart(fig_df)
st.write("Data tail:", data.tail(30))
