import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="AI Aktieanalys PRO", layout="centered")
st.title("📈 Avancerad Aktieanalys i Nuläget")

# Ticker
st.markdown("Skriv in en ticker (t.ex. AAPL, TSLA, VOLV-B.ST):")
ticker = st.text_input("Ticker", value="AAPL").upper()

# Funktion: Förutsäg stängning
@st.cache_data

def predict_close(data):
    today = pd.Timestamp.now(tz="UTC").date()
    intraday = data[data.index.date == today]
    if len(intraday) < 10:
        return None
    X = np.arange(len(intraday)).reshape(-1, 1)
    y = intraday['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    x_future = np.array([[len(intraday) + 5]])
    return float(model.predict(x_future)[0][0])

# Funktion: Candlestick-mönster
candlestick_patterns = {
    "Hammer": lambda o, h, l, c: (h - l) > 3 * (o - c) and (c - l) / (.001 + h - l) > 0.6,
    "Shooting Star": lambda o, h, l, c: (h - l) > 3 * (o - c) and (h - o) / (.001 + h - l) > 0.6,
}

def detect_candles(data):
    patterns = []
    for name, func in candlestick_patterns.items():
        if func(data.Open.iloc[-1], data.High.iloc[-1], data.Low.iloc[-1], data.Close.iloc[-1]):
            patterns.append(name)
    return patterns

if ticker:
    try:
        df = yf.download(ticker, period="5d", interval="1m")
        df.dropna(inplace=True)

        if df.empty:
            st.error("❌ Ingen data hittades. Kontrollera tickern.")
        else:
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            close = df['Close'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            support = df['Close'].rolling(window=50).min().iloc[-1]
            resistance = df['Close'].rolling(window=50).max().iloc[-1]

            candle_hits = detect_candles(df)

            # Risk/Reward-nivåer (enkel 2:1)
            risk = close - support
            reward = resistance - close
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0

            signal = "HÅLL 🤝"
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"

            pred_close = predict_close(df)

            # Visa resultat
            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            st.markdown(f"📊 **Risk/Reward-förhållande:** {rr_ratio}:1")
            if pred_close:
                st.markdown(f"📉 **Förväntad stängning:** {pred_close:.2f} kr")
            if candle_hits:
                st.markdown(f"🕯️ **Candlestick-mönster:** {', '.join(candle_hits)}")

            with st.expander("Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- SMA50: {sma50:.2f}")
                st.write(f"- SMA200: {sma200:.2f}")
                st.write(f"- Volym: {df['Volume'].iloc[-1]}")

            # Candlestick + indikator-graf
            st.subheader("📉 Prisdiagram med indikatorer")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Close'], label='Pris', color='black')
            ax.plot(df['SMA50'], label='SMA50', linestyle='--')
            ax.plot(df['SMA200'], label='SMA200', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f} kr)')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f} kr)')
            ax.set_title(f"{ticker} – Prisdiagram")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Candlestick-graf med volym
            st.subheader("🕯️ Candlestick-graf med volym")
            mpf.plot(df.tail(100), type='candle', volume=True, style='yahoo')

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
