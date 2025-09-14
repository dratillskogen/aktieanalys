import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit layout
st.set_page_config(page_title="Aktieanalysverktyg", layout="wide")
st.title("📈 Enkel Aktieanalys med Signal och Prisnivåer")

# Användarinmatning
ticker = st.text_input("Ange aktiens ticker (t.ex. AAPL, TSLA, VOLV-B.ST):")

if ticker:
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start="2022-01-01", end=today)

    # Indikatorer
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Stöd- och motståndsnivåer (senaste 50 dagar)
    data['Rolling_Min'] = data['Close'].rolling(window=50).min()
    data['Rolling_Max'] = data['Close'].rolling(window=50).max()

    data.dropna(inplace=True)
    latest = data.iloc[-1]

    # Konvertera till float
    def to_float(value):
        if hasattr(value, "item"):
            return value.item()
        elif hasattr(value, "values"):
            return float(value.values[0])
        else:
            return float(value)

    rsi = to_float(latest['RSI'])
    macd = to_float(latest['MACD'])
    macd_signal = to_float(latest['MACD_Signal'])
    close = to_float(latest['Close'])
    sma50 = to_float(latest['SMA50'])
    sma200 = to_float(latest['SMA200'])
    support = to_float(latest['Rolling_Min'])
    resistance = to_float(latest['Rolling_Max'])

    # Signal
    if rsi < 30 and macd < macd_signal:
        signal = "📥 Stark Köp"
    elif rsi > 70 and macd > macd_signal:
        signal = "📤 Stark Sälj"
    else:
        signal = "📊 Håll"

    # Visa resultat
    st.subheader(f"Analys för {ticker.upper()} – {today}")
    st.write(f"**RSI**: {rsi:.2f}")
    st.write(f"**MACD**: {macd:.2f}")
    st.write(f"**MACD-signal**: {macd_signal:.2f}")
    st.write(f"**Stängningspris**: {close:.2f} kr")
    st.write(f"**SMA50**: {sma50:.2f}")
    st.write(f"**SMA200**: {sma200:.2f}")
    st.write(f"**Köp runt**: {support:.2f} kr (stöd)")
    st.write(f"**Sälj runt**: {resistance:.2f} kr (motstånd)")
    st.write(f"**Rekommendation**: {signal}")

    # Prisgraf med stöd/motstånd
    st.subheader("📉 Pris, Glidande Medelvärden & Prisnivåer")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Close'], label='Pris', color='black')
    ax.plot(data['SMA50'], label='SMA50', linestyle='--')
    ax.plot(data['SMA200'], label='SMA200', linestyle='--')
    ax.axhline(support, color='green', linestyle=':', label=f'Stöd ({support:.2f} kr)')
    ax.axhline(resistance, color='red', linestyle=':', label=f'Motstånd ({resistance:.2f} kr)')
    ax.set_title(f"{ticker.upper()} – Prisgraf")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # RSI-graf
    st.subheader("📊 RSI (Relative Strength Index)")
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    ax2.plot(data['RSI'], label='RSI', color='purple')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.set_title("RSI")
    ax2.grid()
    st.pyplot(fig2)
