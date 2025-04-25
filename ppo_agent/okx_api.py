import requests
import pandas as pd
import logging

BASE_URL = 'https://www.okx.com'


class OKXClient:
    def __init__(self):
        self.session = requests.Session()

    def get_historical_candles(self, instrument_id, interval='1h', limit=100):
        endpoint = f'{BASE_URL}/api/v5/market/history-candles'
        params = {
            'instId': instrument_id,
            'bar': interval,
            'limit': limit
        }

        response = self.session.get(endpoint, params=params)

        if response.status_code != 200:
            logging.error(f"Ошибка при запросе данных с OKX: {response.text}")
            return pd.DataFrame()

        data = response.json()

        if 'data' not in data:
            logging.error(f"Некорректный формат данных с OKX: {data}")
            return pd.DataFrame()

        candles = data['data']

        df = pd.DataFrame(candles, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'
        ])

        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.sort_values('time').reset_index(drop=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df