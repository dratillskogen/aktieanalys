# AI-Aktieanalys: Optimerad för Daytrading i Streamlit (mobilvänlig)

import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- SIDKONFIG ---
st.set_page_config(page_title="Daytrading Analys", layout="wide")
st_autorefresh = st.experimental_rerun if st.button("🔄 Uppdatera manuellt") else None

# Automatisk uppdatering var 60:e sekund
countdown = st.empty()
for i in range(60, 0, -1):
    countdown.text(f"🔄 Uppdaterar om {i} sekunder...")
    time.sleep(1)
st.experimental_rerun()
st.title("📈 AI-Aktieanalys för Daytrading")

# --- INPUT ---
ticker = st.text_input("Ange ticker (ex. AAPL, TSLA, VOLV-B.ST):", value="AAPL").upper()
interval = st.selectbox("Välj intervall:", ["1m", "5m", "15m"])

# --- FUNKTION FÖR NUMMER ---
def get_number(val):
    if isinstance(val, pd.Series):
        return val.iloc[-1]
    elif isinstance(val, (np.ndarray, list)):
        return val[-1]
    elif hasattr(val, 'item'):
        return val.item()
    else:
        try:
            return float(val)
        except:
            return np.nan

# --- HÄMTA OCH ANALYSERA DATA ---
if ticker:
    try:
        data = yf.download(ticker, period="5d", interval=interval)
        data = data.dropna()

        if data.empty:
            st.warning("Ingen data hittades. Dubbelkolla tickern.")
        else:
            # TEKNISKA INDIKATORER
            data['EMA9'] = data['Close'].ewm(span=9).mean()
            data['EMA21'] = data['Close'].ewm(span=21).mean()
            data['SMA50'] = data['Close'].rolling(50).mean()
            data['SMA200'] = data['Close'].rolling(200).mean()

            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # SENASTE VÄRDEN
            latest = data.iloc[-1]
            rsi = get_number(latest['RSI'])
            macd = get_number(latest['MACD'])
            macd_signal = get_number(latest['MACD_Signal'])
            close = get_number(latest['Close'])
            sma50 = get_number(latest['SMA50'])
            sma200 = get_number(latest['SMA200'])
            ema9 = get_number(latest['EMA9'])
            ema21 = get_number(latest['EMA21'])

            support = get_number(data['Close'].rolling(50).min().iloc[-1])
            resistance = get_number(data['Close'].rolling(50).max().iloc[-1])

            # FÖRVÄNTAD STÄNGNING
            today = pd.Timestamp.now(tz="UTC").date()
            today_data = data[data.index.date == today]
            if len(today_data) >= 5:
                X = np.arange(len(today_data)).reshape(-1, 1)
                y = today_data['Close'].values.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                pred = model.predict(np.array([[len(today_data)+5]]))[0][0]
            else:
                pred = None

            # SIGNALLOGIK
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # --- UTDATA ---
            st.subheader(f"Signal för {ticker} ({interval})")
            st.markdown(f"### ✅ {signal}")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if pred:
                st.markdown(f"📉 **Förväntad stängning:** {pred:.2f} kr")

            # --- DETALJER ---
            with st.expander("🔍 Visa indikatorer"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- EMA9: {ema9:.2f} | EMA21: {ema21:.2f}")
                st.write(f"- SMA50: {sma50:.2f} | SMA200: {sma200:.2f}")

            # --- PRISDIAGRAM ---
            st.subheader("📊 Prisdiagram med EMA och stöd/motstånd")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['EMA9'], label='EMA9', linestyle='--')
            ax.plot(data['EMA21'], label='EMA21', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f})')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f})')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- VOLYM ---
            st.subheader("📦 Volymanalys")
            fig2, ax2 = plt.subplots(figsize=(10, 2))
            ax2.bar(data.index, data['Volume'], color='gray')
            ax2.set_title("Volym")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
