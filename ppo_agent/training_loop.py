import time
from datetime import datetime
from pathlib import Path

from ppo_agent.agent import PPOAgent
from ppo_agent.indicators import compute_all_indicators
from ppo_agent.features import extract_features
from ppo_agent.enums import SUPPORTED_TIMEFRAMES, INTERVAL_TO_TIMESPAN
from ppo_agent.utils import TrainingJournal
from ppo_agent.asset_registry import get_broker_and_figi
from brokers.broker import get_broker


class PPOTrainingLoop:
    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.journal_root = Path("ppo_agent/training_journal")

    def run(self):
        for asset_name in self.agent.assets:
            broker_name, figi = get_broker_and_figi(asset_name)
            if not broker_name or not figi:
                print(f"⚠️ Актив {asset_name} не найден в asset_registry")
                continue

            broker = get_broker(broker_name)
            if not broker:
                print(f"⚠️ Брокер {broker_name} не зарегистрирован")
                continue

            print(f"🔄 Обработка актива: {asset_name} ({figi}) от {broker_name}")

            for timeframe, interval in SUPPORTED_TIMEFRAMES.items():
                try:
                    steps = 100
                    to_time = datetime.utcnow()
                    from_time = to_time - steps * INTERVAL_TO_TIMESPAN[interval]

                    # Инициализация индивидуального журнала обучения
                    asset_dir = self.journal_root / asset_name
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    journal_file = asset_dir / f"{asset_name}_{timeframe}.json"
                    journal = TrainingJournal(journal_file)

                    if journal.was_trained(figi, timeframe, from_time, to_time):
                        print(f"⏭️ Уже обучено на {asset_name} ({figi}) — {timeframe}")
                        continue

                    candles_df = broker.get_market_data_history(
                        figi=figi,
                        from_=from_time,
                        to_=to_time,
                        interval=interval
                    )

                    if candles_df.empty or len(candles_df) < 52:
                        print(f"⚠️ Недостаточно данных для {asset_name} ({figi}) — {timeframe}")
                        continue

                    features_df = compute_all_indicators(candles_df)
                    if features_df.empty:
                        print(f"⚠️ Не удалось рассчитать признаки для {asset_name} ({figi}) — {timeframe}")
                        continue

                    features_df = extract_features(features_df)
                    features_df["figi"] = figi
                    features_df["timeframe"] = timeframe
                    features_df["timestamp"] = datetime.utcnow()

                    # Обучение
                    self.agent.train_on_dataframe(features_df)

                    # Запись в журнал
                    journal.record_training(figi, timeframe, from_time, to_time)

                    print(f"✅ Дообучен PPO на {asset_name} ({figi}) — {timeframe}")
                    time.sleep(0.5)

                except Exception as e:
                    print(f"❌ Ошибка в обучении {asset_name} ({figi}) — {timeframe}: {e}")