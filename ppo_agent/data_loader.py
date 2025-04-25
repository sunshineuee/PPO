import pandas as pd
from datetime import datetime
from typing import Literal

from ppo_agent.indicators import compute_all_indicators
from brokers.broker import get_broker
from ppo_agent.enums import SUPPORTED_TIMEFRAMES, INTERVAL_TO_TIMESPAN


broker = get_broker("tinkoff")

def load_recent_candles(asset: str, timeframe: Literal["1m", "5m", "15m", "1h", "1d"], steps: int = 100) -> pd.DataFrame:
    """
    Загружает последние свечи и считает индикаторы.
    """
    interval = SUPPORTED_TIMEFRAMES[timeframe]
    interval_duration = INTERVAL_TO_TIMESPAN[interval]
    to_time = datetime.utcnow()
    from_time = to_time - steps * interval_duration

    df = broker.get_market_data_history(
        figi=asset,
        from_=from_time,
        to_=to_time,
        interval=interval
    )

    if df.empty or len(df) < 10:
        raise ValueError(f"Недостаточно свечей для {asset} на таймфрейме {timeframe}")

    return compute_all_indicators(df)


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл и приводит к стандартному формату: asset, timeframe, <features...>
    """
    df = pd.read_csv(csv_path)
    required_columns = {"asset", "timeframe"}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV должен содержать минимум колонки: {required_columns}")

    return df
