import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Aktieanalys", layout="wide")
st.title("📊 AI Aktieanalys – Daglig Handelsvy")

ticker = st.text_input("Ange en aktieticker (t.ex. AAPL, TSLA, VOLV-B.ST):", value="AAPL").upper()

def get_number(val):
    try:
        if isinstance(val, pd.Series):
            return float(val.iloc[-1])
        return float(val)
    except:
        return None

if ticker:
    try:
        data = yf.download(ticker, period="5d", interval="5m")
        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera ticker.")
        else:
            # Tekniska indikatorer
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()

            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Hämta senaste datapunkt
            latest = data.dropna().iloc[-1]
            rsi = get_number(latest['RSI'])
            macd = get_number(latest['MACD'])
            macd_signal = get_number(latest['MACD_Signal'])
            close = get_number(latest['Close'])
            sma50 = get_number(latest['SMA50'])
            sma200 = get_number(latest['SMA200'])

            # Stöd/motstånd
            support = float(data['Close'].rolling(window=50).min().dropna().iloc[-1])
            resistance = float(data['Close'].rolling(window=50).max().dropna().iloc[-1])

            # Förväntad stängningskurs
            today = pd.Timestamp.now(tz="UTC").date()
            today_data = data[data.index.date == today]
            if len(today_data) >= 5:
                X = np.arange(len(today_data)).reshape(-1, 1)
                y = today_data['Close'].values.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                future_x = np.array([[len(today_data) + 5]])
                prediction = float(model.predict(future_x)[0][0])
            else:
                prediction = None

            # Signal
            if rsi is not None and macd is not None and macd_signal is not None:
                if rsi < 30 and macd < macd_signal:
                    signal = "KÖP 📥"
                elif rsi > 70 and macd > macd_signal:
                    signal = "SÄLJ 📤"
                else:
                    signal = "HÅLL 🤝"
            else:
                signal = "Ingen signal"

            # Visa resultat
            st.subheader(f"Signal för {ticker} – Senaste datan")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if prediction:
                st.markdown(f"📉 **Förväntad stängning:** ca {prediction:.2f} kr")

            with st.expander("🔍 Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Close: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")

            # Fibonacci-nivåer
            fib_low = float(data['Low'].dropna().min())
            fib_high = float(data['High'].dropna().max())
            fib_levels = [fib_high - (fib_high - fib_low) * level for level in [0.236, 0.382, 0.5, 0.618, 0.786]]

            # Pris + volym + fib-graf
            st.subheader("📈 Pris- och volymdiagram med Fibonacci-nivåer")
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(data['Close'], label='Stängningspris', color='black')
            ax1.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax1.plot(data['SMA200'], label='SMA200', linestyle='--')
            for lvl in fib_levels:
                ax1.axhline(y=lvl, linestyle='dotted', color='blue', alpha=0.5)
            ax1.set_ylabel("Pris")
            ax1.legend(loc="upper left")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.bar(data.index, data['Volume'], width=0.01, color='grey', alpha=0.3)
            ax2.set_ylabel("Volym")

            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Ett fel uppstod: {e}")
