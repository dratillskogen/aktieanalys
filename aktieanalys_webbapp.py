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
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, pd.Series):
        return val.values[0]
    else:
        return float(val)

if ticker:
    try:
        data = yf.download(ticker, period="5d", interval="5m")
        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera ticker.")
        else:
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

            latest = data.iloc[-1]
            rsi = get_number(latest['RSI'])
            macd = get_number(latest['MACD'])
            macd_signal = get_number(latest['MACD_Signal'])
            close = get_number(latest['Close'])
            sma50 = get_number(latest['SMA50'])
            sma200 = get_number(latest['SMA200'])

            # Fibonacci nivåer (support/resistance)
            fib_low = get_number(data['Low'].min())
            fib_high = get_number(data['High'].max())
            fib_levels = [fib_high - (fib_high - fib_low) * level for level in [0.236, 0.382, 0.5, 0.618, 0.786]]
            support = get_number(data['Close'].rolling(window=50).min().iloc[-1])
            resistance = get_number(data['Close'].rolling(window=50).max().iloc[-1])

            # Förväntad stängning
            today = pd.Timestamp.now(tz="UTC").date()
            today_data = data[data.index.date == today]
            if len(today_data) >= 5:
                X = np.arange(len(today_data)).reshape(-1, 1)
                y = today_data['Close'].values.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                future_x = np.array([[len(today_data) + 5]])
                prediction = model.predict(future_x)[0][0]
            else:
                prediction = None

            # SIGNAL - FÖRÄNDRADE TILL float jämförelse
            if float(rsi) < 30 and float(macd) < float(macd_signal):
                signal = "KÖP 📥"
            elif float(rsi) > 70 and float(macd) > float(macd_signal):
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # Visa info
            st.subheader(f"Signal för {ticker} – Senaste datan")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if prediction:
                st.markdown(f"📉 **Förväntad stängning:** ca {prediction:.2f} kr")

            # Detaljer
            with st.expander("🔍 Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")

            # Pris + Volym-graf med Fibonacci
            st.subheader("📈 Pris & volymdiagram med Fibonacci-nivåer")
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(data['Close'], label='Stängningspris', color='black')
            ax1.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax1.plot(data['SMA200'], label='SMA200', linestyle='--')
            for lvl in fib_levels:
                ax1.axhline(lvl, linestyle='dotted', color='blue', alpha=0.5)
            ax1.set_ylabel("Pris")
            ax1.legend(loc="upper left")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.bar(data.index, data['Volume'], width=0.01, color='grey', alpha=0.3)
            ax2.set_ylabel("Volym")

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
