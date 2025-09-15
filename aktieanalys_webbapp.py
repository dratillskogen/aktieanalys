import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="AI Aktieanalys", layout="centered")
st.title("📈 Aktieanalys i Nuläget")

# 1. Ticker-input
st.markdown("Skriv in en ticker (t.ex. AAPL, TSLA, VOLV-B.ST):")
ticker = st.text_input("Ticker", value="AAPL").upper()

if ticker:
    try:
        # 2. Hämta den senaste datan
        data = yf.download(ticker, period="5d", interval="1m")
        data = data.dropna()

        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera att tickern är korrekt.")
        else:
            # 3. Teknisk analys
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
            rsi = latest['RSI'].item() if hasattr(latest['RSI'], 'item') else latest['RSI']
            macd = latest['MACD'].item() if hasattr(latest['MACD'], 'item') else latest['MACD']
            macd_signal = latest['MACD_Signal'].item() if hasattr(latest['MACD_Signal'], 'item') else latest['MACD_Signal']
            close = latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close']

            # 4. Enkel logik för signal
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # 5. Visa signal
            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")

            # 6. Expander för detaljerad analys
            with st.expander("Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {latest['SMA50']:.2f} kr")
                st.write(f"- SMA200: {latest['SMA200']:.2f} kr")

            # 7. Graf
            st.subheader("Prisdiagram")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax.plot(data['SMA200'], label='SMA200', linestyle='--')
            ax.set_title(f"{ticker} – Senaste priset")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
