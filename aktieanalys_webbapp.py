import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import mplfinance as mpf

st.set_page_config(page_title="AI Aktieanalys", layout="wide")
st.title("📊 AI Aktieanalys – Daglig Handelsvy")

# --- Ticker input ---
ticker = st.text_input("Ange en aktieticker (t.ex. AAPL, TSLA, VOLV-B.ST):", value="AAPL").upper()

if ticker:
    try:
        # --- Hämta data ---
        data = yf.download(ticker, period="5d", interval="5m")
        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera ticker.")
        else:
            # --- Indikatorer ---
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

            # --- Senaste värden ---
            latest = data.iloc[-1]
            def get_number(val):
                if hasattr(val, 'item'): return val.item()
                elif isinstance(val, pd.Series): return val.values[0]
                else: return float(val)

            rsi = get_number(latest['RSI'])
            macd = get_number(latest['MACD'])
            macd_signal = get_number(latest['MACD_Signal'])
            close = get_number(latest['Close'])
            sma50 = get_number(latest['SMA50'])
            sma200 = get_number(latest['SMA200'])

            # --- Stöd/motstånd ---
            support = get_number(data['Close'].rolling(window=50).min().iloc[-1])
            resistance = get_number(data['Close'].rolling(window=50).max().iloc[-1])

            # --- Förväntad stängningskurs ---
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

            # --- Signal ---
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # --- Visa signal ---
            st.subheader(f"Signal för {ticker} – Senaste datan")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if prediction:
                st.markdown(f"📉 **Förväntad stängning:** ca {prediction:.2f} kr")

            # --- Expander för analys ---
            with st.expander("🔍 Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")

            # --- Candlestick-graf + volym + Fibonacci ---
            st.subheader("📉 Candlestick med volym och Fibonacci")
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()
            df = df.astype(float)
            df.index.name = 'Date'
            df = df[-100:]

            fib_low = df['Low'].min()
            fib_high = df['High'].max()
            fib_levels = [fib_high - (fib_high - fib_low) * level for level in [0.236, 0.382, 0.5, 0.618, 0.786]]

            fib_addplots = [mpf.make_addplot([lvl]*len(df), color='blue', linestyle='dotted') for lvl in fib_levels]
            mpf_fig, _ = mpf.plot(df, type='candle', volume=True, addplot=fib_addplots, returnfig=True, style='yahoo')
            st.pyplot(mpf_fig)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
