import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 Daglig Aktieanalys", layout="centered")
st.title("📈 Realtidsbaserad Aktieanalys")

# 🟦 Input
ticker = st.text_input("Skriv en ticker (t.ex. AAPL, TSLA, VOLV-B.ST)", "AAPL").upper()

# 🧠 Funktion: Förväntad stängningskurs
def förväntad_stängning(data):
    today = pd.Timestamp.now(tz="UTC").date()
    today_data = data[data.index.date == today]

    if len(today_data) < 10:
        return None

    X = np.arange(len(today_data)).reshape(-1, 1)
    y = today_data["Close"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    x_future = np.array([[len(today_data) + 5]])
    return model.predict(x_future)[0][0]

# 📦 Datahämtning
if ticker:
    try:
        data = yf.download(ticker, period="2d", interval="1m", progress=False).dropna()

        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera tickern.")
        else:
            # Glidande medelvärden
            data["SMA50"] = data["Close"].rolling(window=50).mean()
            data["SMA200"] = data["Close"].rolling(window=200).mean()

            # RSI
            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = ema12 - ema26
            data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

            # Stöd & motstånd
            support = data["Close"].rolling(60).min().iloc[-1]
            resistance = data["Close"].rolling(60).max().iloc[-1]

            # Fibonacci retracement
            high = data["Close"].max()
            low = data["Close"].min()
            fib_levels = [high - (high - low) * level for level in [0.236, 0.382, 0.5, 0.618, 0.786]]

            # Senaste datapunkt
            latest = data.iloc[-1]
            rsi = latest["RSI"]
            macd = latest["MACD"]
            macd_signal = latest["MACD_Signal"]
            close = latest["Close"]
            sma50 = latest["SMA50"]
            sma200 = latest["SMA200"]

            # Enkel signal
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # Förutsägelse
            stängning = förväntad_stängning(data)

            # ✅ Visa analys
            st.subheader(f"Signal för {ticker}")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if stängning:
                st.markdown(f"📉 **Förväntad stängningskurs:** {stängning:.2f} kr")

            with st.expander("📋 Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")

            # 📊 Prisgraf
            st.subheader("📉 Pris och indikatorer")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data["Close"], label="Pris", color="black")
            ax.plot(data["SMA50"], label="SMA50", linestyle="--")
            ax.plot(data["SMA200"], label="SMA200", linestyle="--")
            ax.axhline(support, color="green", linestyle=":", label=f"Stöd ({support:.2f})")
            ax.axhline(resistance, color="red", linestyle=":", label=f"Motstånd ({resistance:.2f})")
            for level in fib_levels:
                ax.axhline(level, color="blue", linestyle="--", alpha=0.3)
            ax.set_title(f"{ticker} – Pris och tekniska nivåer")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
