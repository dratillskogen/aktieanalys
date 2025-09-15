import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="📊 Avancerad Aktieanalys", layout="centered")
st.title("📈 Teknisk Aktieanalys i Nuläget")

# 1. Användarens ticker-input
st.markdown("Skriv in en ticker (ex: AAPL, TSLA, VOLV-B.ST):")
ticker = st.text_input("Ticker", value="AAPL").upper()

if ticker:
    try:
        # 2. Hämta data (senaste 5 dagarna, 1-minut intervall)
        data = yf.download(ticker, period="5d", interval="1m")
        data.dropna(inplace=True)

        if data.empty:
            st.error("❌ Ingen data hittades. Kontrollera tickern.")
        else:
            # 3. Indikatorer: SMA, EMA, RSI, MACD
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

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

            # 4. Candlestick-mönster (enkelt: bullish engulfing)
            data['bullish_engulfing'] = (
                (data['Close'] > data['Open']) &
                (data['Close'].shift(1) < data['Open'].shift(1)) &
                (data['Close'] > data['Open'].shift(1)) &
                (data['Open'] < data['Close'].shift(1))
            )

            # 5. Volymanalys
            data['Volym_SMA20'] = data['Volume'].rolling(window=20).mean()

            # 6. Fibonacci Retracement-nivåer (sista 100 datapunkter)
            recent = data.tail(100)
            max_price = recent['High'].max()
            min_price = recent['Low'].min()
            diff = max_price - min_price
            fib_levels = [
                max_price,
                max_price - 0.236 * diff,
                max_price - 0.382 * diff,
                max_price - 0.5 * diff,
                max_price - 0.618 * diff,
                min_price
            ]

            # 7. Uträkning av signalvärden
            latest = data.iloc[-1]
            def safe(val):
                return val.item() if hasattr(val, 'item') else val

            rsi = safe(latest['RSI'])
            macd = safe(latest['MACD'])
            macd_signal = safe(latest['MACD_Signal'])
            close = safe(latest['Close'])
            sma50 = safe(latest['SMA50'])
            sma200 = safe(latest['SMA200'])
            volume = safe(latest['Volume'])
            volume_avg = safe(latest['Volym_SMA20'])

            support = safe(data['Close'].rolling(window=50).min().iloc[-1])
            resistance = safe(data['Close'].rolling(window=50).max().iloc[-1])

            # 8. Signal
            if rsi < 30 and macd < macd_signal:
                signal = "KÖP 📥"
            elif rsi > 70 and macd > macd_signal:
                signal = "SÄLJ 📤"
            else:
                signal = "HÅLL 🤝"

            # 9. Visa analys
            st.subheader(f"Signal för {ticker} (senaste datan)")
            st.markdown(f"### ✅ **{signal}**")
            st.markdown(f"💰 **Köp runt:** {support:.2f} kr")
            st.markdown(f"💸 **Sälj runt:** {resistance:.2f} kr")

            # 10. Detaljerad analys
            with st.expander("📊 Visa detaljerad analys"):
                st.write(f"- RSI: {rsi:.2f}")
                st.write(f"- MACD: {macd:.2f}")
                st.write(f"- MACD Signal: {macd_signal:.2f}")
                st.write(f"- Stängningspris: {close:.2f} kr")
                st.write(f"- SMA50: {sma50:.2f} kr")
                st.write(f"- SMA200: {sma200:.2f} kr")
                st.write(f"- Volym: {volume:.0f} | Snittvolym 20: {volume_avg:.0f}")

            # 11. Plot
            st.subheader("📈 Prisdiagram med indikatorer")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['Close'], label='Pris', color='black')
            ax.plot(data['SMA50'], label='SMA50', linestyle='--')
            ax.plot(data['SMA200'], label='SMA200', linestyle='--')
            ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f})')
            ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f})')
            for level in fib_levels:
                ax.axhline(level, linestyle=':', alpha=0.4, color='blue')
            ax.set_title(f"{ticker} – Teknisk analys")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # 12. Candlestick mönster
            if data['bullish_engulfing'].iloc[-1]:
                st.info("📈 Bullish Engulfing-mönster upptäckt!")

    except Exception as e:
        st.error(f"Ett fel uppstod: {e}")
