from datetime import timedelta
from enum import Enum


class Timeframe(Enum):
    MIN_1 = '1m'
    MIN_5 = '5m'
    MIN_15 = '15m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'


TIMEFRAME_TO_TIMESPAN = {
    Timeframe.MIN_1: timedelta(minutes=1),
    Timeframe.MIN_5: timedelta(minutes=5),
    Timeframe.MIN_15: timedelta(minutes=15),
    Timeframe.HOUR_1: timedelta(hours=1),
    Timeframe.HOUR_4: timedelta(hours=4),
    Timeframe.DAY_1: timedelta(days=1),
}


def get_timespan(timeframe: Timeframe) -> timedelta:
    """
    Возвращает объект timedelta для заданного таймфрейма.
    """
    return TIMEFRAME_TO_TIMESPAN.get(timeframe, timedelta(minutes=1))


def all_timeframes() -> list[Timeframe]:
    """
    Возвращает список всех поддерживаемых таймфреймов.
    """
    return list(Timeframe)
