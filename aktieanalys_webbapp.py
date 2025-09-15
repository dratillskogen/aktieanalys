import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="AI Aktieanalys", layout="centered")
st.title("📈 Aktieanalys i Nuläget")

# 1. Ticker-input
st.markdown("Skriv in en ticker (t.ex. AAPL, TSLA, VOLV-B.ST):")
ticker = st.text_input("Ticker", value="AAPL").upper()

# Funktion för att förutsäga stängningskurs
def förväntad_stängning(data):
    today = pd.Timestamp.now(tz="UTC").date()
    today_data = data[data.index.date == today]
    
    if len(today_data) < 5:
        return None

    X = np.arange(len(today_data)).reshape(-1, 1)
    y = today_data["Close"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    x_future = np.array([[len(today_data) + 5]])
    predicted_price = model.predict(x_future)[0][0]
    return predicted_price

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

            def get_number(val):
                if hasattr(val, 'item'):
                    return val.item()
                elif isinstance(val, pd.Series):
                    return val.values[0]
                else:
                    return float(val)

            rsi = get_number(latest['RSI'])
            macd = get_number(latest['MACD'])
            macd_signal = get_number(latest['MACD_Signal'])
            close = get_number(latest['Close'])
            sma50 = get_number(latest['SMA50'])
            sma200 = get_number(latest['SMA200'])

            # 4. Stöd- och motståndsnivåer (baserat på tidigare lägsta/högsta)
            support = get_number(data['Close'].rolling(window=50).min().iloc[-1])
            resistance = get_number(data['Close'].rolling(window=50).max().iloc[-1])

            # 5. Enkel logik för signal
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # 6. Visa signal
            stängningsprognos = förväntad_stängning(data)
            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if stängningsprognos:
                st.markdown(f"📉 **Förväntad stängningskurs:** ca **{stängningsprognos:.2f} kr**")
            else:
                st.markdown("🔍 Ingen stängningsprognos (för lite dagsdata).")

            # 7. Expander för detaljerad analys
            with st.expander("Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")

            # 8. Graf
            st.subheader("Prisdiagram")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax.plot(data['SMA200'], label='SMA200', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f} kr)')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f} kr)')
            ax.set_title(f"{ticker} – Senaste priset")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
