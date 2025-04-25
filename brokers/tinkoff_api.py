from datetime import datetime, timedelta
import pandas as pd
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now

class TinkoffAPI:
    def __init__(self, token: str):
        self.token = token

    def get_candles(
        self,
        figi: str,
        from_: datetime,
        to_: datetime,
        interval: CandleInterval
    ) -> pd.DataFrame:
        candles = []

        try:
            with Client(self.token) as client:
                raw_candles = list(client.get_all_candles(
                    figi=figi,
                    from_=from_,
                    to=to_,
                    interval=interval
                ))

                for c in raw_candles:
                    candles.append({
                        "time": c.time,
                        "open": c.open.units + c.open.nano / 1e9,
                        "high": c.high.units + c.high.nano / 1e9,
                        "low": c.low.units + c.low.nano / 1e9,
                        "close": c.close.units + c.close.nano / 1e9,
                        "volume": c.volume
                    })

        except Exception as e:
            print(f"❌ Ошибка получения свечей по {figi}: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df.set_index("time", inplace=True)
        return df

    def get_latest_candles(
        self,
        figi: str,
        interval: CandleInterval,
        steps: int
    ) -> pd.DataFrame:
        interval_seconds = self._get_interval_seconds(interval)
        to_ = datetime.utcnow()
        from_ = to_ - timedelta(seconds=steps * interval_seconds)
        return self.get_candles(figi, from_, to_, interval)

    def _get_interval_seconds(self, interval: CandleInterval) -> int:
        if interval == CandleInterval.CANDLE_INTERVAL_1_MIN:
            return 60
        elif interval == CandleInterval.CANDLE_INTERVAL_5_MIN:
            return 300
        elif interval == CandleInterval.CANDLE_INTERVAL_15_MIN:
            return 900
        elif interval == CandleInterval.CANDLE_INTERVAL_HOUR:
            return 3600
        elif interval == CandleInterval.CANDLE_INTERVAL_DAY:
            return 86400
        elif interval == CandleInterval.CANDLE_INTERVAL_WEEK:
            return 604800
        elif interval == CandleInterval.CANDLE_INTERVAL_MONTH:
            return 2592000
        else:
            raise ValueError(f"Неизвестный интервал свечей: {interval}")
