
from pathlib import Path

import time
from datetime import datetime

from typing import Dict

from ppo_agent.enums import SUPPORTED_TIMEFRAMES, INTERVAL_TO_TIMESPAN

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from ppo_agent.data_loader import load_recent_candles
from ppo_agent.indicators import compute_all_indicators
from ppo_agent.enums import SUPPORTED_TIMEFRAMES
from ppo_agent.features import extract_features

from ppo_agent.utils import TrainingJournal

class PPOAgent:
    def __init__(self, model_dir: str = "ppo_agent/models", journal_dir: str = "training_journal"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, PPO] = {}
        self.env = DummyVecEnv([lambda: make_vec_env("CartPole-v1", n_envs=1)])
        self.journal = TrainingJournal(journal_dir)
        self.load_all_models()

    def train_from_dataframe(self, df: pd.DataFrame):
        grouped = df.groupby(['figi', 'timeframe'])
        for (figi, timeframe), group in grouped:
            if not figi or timeframe not in SUPPORTED_TIMEFRAMES:
                continue

            from_time = pd.to_datetime(group["timestamp"].min())
            to_time = pd.to_datetime(group["timestamp"].max())

            # Проверка по журналу
            if self.journal.was_trained(figi, timeframe, from_time, to_time):
                print(f"⏭️ Пропуск обучения: {figi} {timeframe} уже обучен в интервале {from_time} - {to_time}")
                continue

            features = extract_features(group)
            if 'reward' not in group.columns:
                print(f"⚠️ Нет колонки reward: {figi} {timeframe}")
                continue

            observations = features.to_numpy()
            rewards = group['reward'].to_numpy()

            if len(observations) < 10:
                print(f"⚠️ Недостаточно данных для обучения: {figi} {timeframe}")
                continue

            print(f"🧠 Обучение {figi} {timeframe} | примеров: {len(observations)} | reward avg: {rewards.mean():.4f}")

            model = PPO("MlpPolicy", self.env, verbose=0)
            model.learn(total_timesteps=len(observations))
            model_path = self.model_dir / f"{figi}_{timeframe}.zip"
            model.save(model_path)
            self.models[f"{figi}_{timeframe}"] = model

            self.journal.record_training(figi, timeframe, from_time, to_time)

    def self_training_loop(self, interval_sec: float = 0.5):
        while True:
            for key in list(self.models.keys()):
                try:
                    figi, timeframe = key.rsplit("_", 1)
                    interval = SUPPORTED_TIMEFRAMES.get(timeframe)
                    if not interval:
                        continue

                    to_time = datetime.utcnow()
                    from_time = to_time - INTERVAL_TO_TIMESPAN[interval] * 100

                    # Проверка, обучался ли уже этот интервал
                    if self.journal.was_trained(figi, timeframe, from_time, to_time):
                        print(f"⏭️ Пропуск {figi} {timeframe}: уже обучено за {from_time} — {to_time}")
                        continue

                    df = load_recent_candles(figi, timeframe, 100)
                    if df.empty or len(df) < 52:
                        print(f"⚠️ Недостаточно свечей: {figi} {timeframe}")
                        continue

                    features_df = compute_all_indicators(df)
                    if 'reward' not in features_df.columns:
                        print(f"⚠️ Нет reward: {figi} {timeframe}")
                        continue

                    features_df = extract_features(features_df)
                    observations = features_df.to_numpy()
                    rewards = features_df["reward"].to_numpy()

                    model = self.models[key]
                    model.learn(total_timesteps=len(observations))
                    model.save(self.model_dir / f"{figi}_{timeframe}.zip")

                    self.journal.record_training(figi, timeframe, from_time, to_time)
                    print(f"✅ Дообучено: {figi} {timeframe} | {len(observations)} примеров")

                except Exception as e:
                    print(f"❌ Ошибка в self_training {key}: {e}")

                time.sleep(interval_sec)
