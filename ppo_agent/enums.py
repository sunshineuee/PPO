from datetime import timedelta
from tinkoff.invest import CandleInterval

# Поддерживаемые таймфреймы и их строковые идентификаторы
SUPPORTED_TIMEFRAMES = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

# Преобразование таймфрейма в timedelta
INTERVAL_TO_TIMESPAN = {
    CandleInterval.CANDLE_INTERVAL_1_MIN: timedelta(minutes=1),
    CandleInterval.CANDLE_INTERVAL_5_MIN: timedelta(minutes=5),
    CandleInterval.CANDLE_INTERVAL_15_MIN: timedelta(minutes=15),
    CandleInterval.CANDLE_INTERVAL_HOUR: timedelta(hours=1),
    CandleInterval.CANDLE_INTERVAL_DAY: timedelta(days=1),
    CandleInterval.CANDLE_INTERVAL_WEEK: timedelta(weeks=1),
    CandleInterval.CANDLE_INTERVAL_MONTH: timedelta(days=30),  # Упрощение
}
