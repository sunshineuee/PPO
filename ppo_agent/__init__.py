# ppo_agent/__init__.py

from pathlib import Path

# Каталоги по умолчанию
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
JOURNAL_DIR = Path(__file__).parent / "training_journal"

# Убедимся, что каталоги существуют
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
