from pydantic import BaseModel


class PredictionResult(BaseModel):
    asset: str               # Название актива (например, BTC-USD)
    timeframe: str           # Таймфрейм (например, "15m", "1h", "1d")
    direction: float         # Сила сигнала (направление, от -1 до 1)
    confidence: float        # Уверенность в прогнозе (0.0 - 1.0)
