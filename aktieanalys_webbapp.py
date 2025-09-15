import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Konfiguration
st.set_page_config(page_title="AI Aktieanalys", layout="centered")
st.title("📈 Avancerad Aktieanalys i Nuläget")

# Input för ticker
ticker = st.text_input("Skriv in en ticker (t.ex. AAPL, TSLA, VOLV-B.ST):", value="AAPL").upper()

if ticker:
    try:
        # Hämta data (1-minutersintervall för senaste 5 dagarna)
        data = yf.download(ticker, period="5d", interval="1m")
        data = data.dropna()

        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera att tickern är korrekt.")
        else:
            # Teknisk analys
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

            # Volymförändring
            data['Volymförändring'] = data['Volume'].pct_change() * 100

            # Förväntad stängningskurs (linjär regression)
            today_data = data[data.index.date == pd.Timestamp.now(tz="UTC").date()]
            stängningsprognos = None
            if len(today_data) > 5:
                X = np.arange(len(today_data)).reshape(-1, 1)
                y = today_data['Close'].values.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                stängningsprognos = model.predict([[len(today_data) + 5]])[0][0]

            # Candlestick-mönster (enkelt exempel: Hammer)
            data['Body'] = data['Close'] - data['Open']
            data['Range'] = data['High'] - data['Low']
            data['Hammer'] = (data['Body'].abs() < 0.3 * data['Range']) & ((data['Close'] - data['Low']) > 2 * data['Body'].abs())

            # Stöd/motståndsnivåer
            support = data['Close'].rolling(window=50).min().iloc[-1]
            resistance = data['Close'].rolling(window=50).max().iloc[-1]

            # Senaste värden
            latest = data.iloc[-1]
            rsi = latest['RSI'].item() if hasattr(latest['RSI'], 'item') else latest['RSI']
            macd = latest['MACD'].item() if hasattr(latest['MACD'], 'item') else latest['MACD']
            macd_signal = latest['MACD_Signal'].item() if hasattr(latest['MACD_Signal'], 'item') else latest['MACD_Signal']
            close = latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close']
            sma50 = latest['SMA50'].item() if hasattr(latest['SMA50'], 'item') else latest['SMA50']
            sma200 = latest['SMA200'].item() if hasattr(latest['SMA200'], 'item') else latest['SMA200']

            # Signal-logik
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # Visa signal
            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")
            if stängningsprognos:
                st.markdown(f"📉 **Förväntad stängningskurs:** ca **{stängningsprognos:.2f} kr**")

            with st.expander("Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")
                st.write(f"- Volymförändring (senaste): {latest['Volymförändring']:.2f}%")
                if latest['Hammer']:
                    st.write("- Candlestick-mönster: 🔨 Hammer upptäckt")

            # Prisgraf + nivåer
            st.subheader("Prisdiagram")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax.plot(data['SMA200'], label='SMA200', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f})')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f})')
            ax.set_title(f"{ticker} – Senaste priset")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Volymdiagram
            st.subheader("Volym")
            fig2, ax2 = plt.subplots(figsize=(10, 2))
            ax2.bar(data.index[-200:], data['Volume'].iloc[-200:], width=0.005, color='gray')
            ax2.set_title("Volym senaste tiden")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
