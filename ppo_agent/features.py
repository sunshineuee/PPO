import pandas as pd
from typing import List, Dict
import ta  # Technical Analysis library


# Соответствия между названиями колонок в CSV и используемыми признаками модели
FEATURE_MAPPING: Dict[str, str] = {
    "rsi": "rsi",
    "atr": "atr",
    "macd": "macd",
    "ema12": "ema12",
    "ema26": "ema26",
    "volume": "volume",
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "bb_upper": "bb_upper",
    "bb_lower": "bb_lower",
    "adx": "adx",
    "cci": "cci",
    "willr": "willr"
}


def extract_features(df: pd.DataFrame, required_only: bool = False) -> pd.DataFrame:
    """
    Преобразует таблицу данных, оставляя только валидные признаки, переименовывает их, заполняет пропуски.
    """
    features = []
    for col in df.columns:
        if col.lower() in FEATURE_MAPPING:
            features.append((col, FEATURE_MAPPING[col.lower()]))

    if required_only and not features:
        raise ValueError("Нет допустимых признаков в данных!")

    feature_df = df[[orig for orig, _ in features]].copy()
    feature_df.columns = [new for _, new in features]

    # Заполняем пропуски и нормализуем (по желанию — тут простая нормализация)
    feature_df = feature_df.fillna(method="ffill").fillna(method="bfill")
    feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)

    return feature_df


def get_feature_names() -> List[str]:
    """
    Возвращает список всех признаков, которые модель может использовать.
    """
    return list(FEATURE_MAPPING.values())

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет технические индикаторы из исходных OHLCV-данных.
    :param df: DataFrame с колонками open, high, low, close, volume
    :return: DataFrame с рассчитанными признаками
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Для вычисления индикаторов требуется колонка 'close'")

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()

    # MACD
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["ema12"] = macd.ema12()
    df["ema26"] = macd.ema26()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # ADX
    df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()

    # CCI
    df["cci"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"]).cci()

    # Williams %R
    df["willr"] = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"]).williams_r()

    # Очистка и нормализация
    df = df.dropna()
    return extract_features(df)
