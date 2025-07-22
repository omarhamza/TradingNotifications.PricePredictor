import requests
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import ta  # pip install ta
from ta.trend import MACD
import joblib
from send_telegram_message import send_df_via_telegram
from config import features

# === PARAMÈTRES ===
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
LIMIT = 1000
DAYS = 365
SEQ_LEN = 60
PRED_HOURS = 48
API_URL = "https://api.binance.com/api/v3/klines"
MODEL_PATH = "eth_model.keras"
SCALER_PATH = "eth_scaler.pkl"


def fetch_binance_data(symbol, interval, days):
    limit = 1000
    df_all = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    while start_time < end_time:
        url = f"{API_URL}?symbol={symbol}&interval={interval}&limit={limit}&startTime={start_time}"
        res = requests.get(url)
        data = res.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df_all.append(df)

        last_time = int(data[-1][0])
        start_time = last_time + 1

    df_final = pd.concat(df_all)
    df_final = df_final[~df_final.index.duplicated(keep='last')]
    return df_final


def add_indicators(df):
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    macd_indicator = MACD(close=df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()

    df.dropna(inplace=True)
    return df


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len][0])  # close price
    return np.array(X), np.array(y)


def train_and_predict(df):
    raw = df[features].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)

    X, _ = create_sequences(scaled, SEQ_LEN)

    y_real = df["close"].values[SEQ_LEN:]
    y_scaled = scaler.fit_transform(y_real.reshape(-1, 1))

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(SEQ_LEN, X.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y_scaled, epochs=10, batch_size=32)

    # === Prédictions sur les dernières séquences ===
    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, scaled.shape[1])
    future_predictions = []

    current_seq = last_seq.copy()
    for _ in range(PRED_HOURS):
        pred_scaled = model.predict(current_seq)[0][0]
        future_predictions.append(pred_scaled)

        next_input = np.hstack([pred_scaled] + [current_seq[0, -1, 1:]])
        current_seq = np.append(current_seq[:, 1:], [[next_input]], axis=1)

    # Inverser le scaling uniquement pour 'close'
    dummy_input = np.zeros((len(future_predictions), scaled.shape[1]))
    dummy_input[:, 0] = future_predictions
    preds_real = scaler.inverse_transform(dummy_input)[:, 0]

    # === Timestamps futurs ===
    last_ts = df.index[-1]
    future_times = [last_ts + pd.Timedelta(hours=i + 1) for i in range(PRED_HOURS)]

    df_pred = pd.DataFrame({
        'timestamp': future_times,
        'predicted_close': preds_real
    })
    df_pred['timestamp'] = df_pred['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-2')
    df_pred.to_csv("eth_predictions.csv", index=False)
    send_df_via_telegram(df_pred)
    print("✅ Prédictions enregistrées dans eth_predictions.csv")

    # Sauvegardes
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("✅ Modèle et scalers sauvegardés.")


print("📦 Téléchargement des données ETH...")
df = fetch_binance_data(SYMBOL, INTERVAL, DAYS)
df = add_indicators(df)
print("📈 Entraînement et prédiction...")
train_and_predict(df)
