import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📈 Aktieanalys med AI", layout="centered")
st.title("🔍 AI-Baserad Aktieanalys i Realtid")

# --- Ticker-input ---
ticker = st.text_input("Skriv in en ticker (t.ex. AAPL, TSLA, VOLV-B.ST):", value="AAPL").upper()

# --- Funktion: Förväntad stängningskurs ---
def förväntad_stängning(data):
    today = pd.Timestamp.now(tz="UTC").date()
    today_data = data[data.index.date == today]
    if len(today_data) < 5:
        return None
    X = np.arange(len(today_data)).reshape(-1, 1)
    y = today_data['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    x_future = np.array([[len(today_data) + 5]])
    return model.predict(x_future)[0][0]

# --- Funktion: Fibonacci-nivåer ---
def fibonacci_levels(data):
    high = data['Close'].rolling(window=50).max().iloc[-1]
    low = data['Close'].rolling(window=50).min().iloc[-1]
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '1.0': low
    }
    return levels

if ticker:
    try:
        data = yf.download(ticker, period="5d", interval="1m")
        data = data.dropna()

        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera tickern.")
        else:
            data['SMA50'] = data['Close'].rolling(50).mean()
            data['SMA200'] = data['Close'].rolling(200).mean()
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['Volume_SMA20'] = data['Volume'].rolling(window=20).mean()

            latest = data.iloc[-1]
            rsi = float(latest['RSI'])
            macd = float(latest['MACD'])
            macd_signal = float(latest['MACD_Signal'])
            close = float(latest['Close'])
            sma50 = float(latest['SMA50'])
            sma200 = float(latest['SMA200'])
            volume = int(latest['Volume'])

            support = float(data['Close'].rolling(window=50).min().iloc[-1])
            resistance = float(data['Close'].rolling(window=50).max().iloc[-1])
            fib_levels = fibonacci_levels(data)

            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            prognos = förväntad_stängning(data)
            if prognos:
                st.markdown(f"📉 **Förväntad stängningskurs:** {prognos:.2f} kr")

            with st.expander("Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")
                st.write(f"- Volym: {volume}")

            st.subheader("📊 Prisdiagram med indikatorer")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax.plot(data['SMA200'], label='SMA200', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support} kr)')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance} kr)')
            for level, price in fib_levels.items():
                ax.axhline(price, linestyle='--', alpha=0.3, label=f'Fib {level}')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("📉 Candlestick-graf med volym")
            mpf_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            mpf_data.index.name = 'Date'
            mpf.plot(mpf_data, type='candle', volume=True, style='yahoo', mav=(50, 200))

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
