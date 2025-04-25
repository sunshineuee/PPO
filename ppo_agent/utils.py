from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json
from filelock import FileLock


class TrainingJournal:
    def __init__(self, root_dir: str = "training_journal"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_path(self, figi: str, timeframe: str) -> Path:
        dir_path = self.root / figi
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{figi}_{timeframe}.json"

    def _get_lock(self, path: Path) -> FileLock:
        return FileLock(str(path) + ".lock")

    def _load(self, path: Path) -> list:
        if path.exists():
            with self._get_lock(path), open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self, path: Path, data: list):
        with self._get_lock(path), open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def was_trained(self, figi: str, timeframe: str, from_: datetime, to_: datetime) -> bool:
        path = self._get_path(figi, timeframe)
        entries = self._load(path)
        for entry in entries:
            trained_from = datetime.fromisoformat(entry["from"])
            trained_to = datetime.fromisoformat(entry["to"])
            if trained_from <= from_ and trained_to >= to_:
                return True
        return False

    def record_training(self, figi: str, timeframe: str, from_: datetime, to_: datetime):
        path = self._get_path(figi, timeframe)
        entries = self._load(path)
        entries.append({
            "from": from_.isoformat(),
            "to": to_.isoformat()
        })
        self._save(path, entries)

    def get_all_intervals(self, figi: str) -> Dict[str, list]:
        figi_dir = self.root / figi
        if not figi_dir.exists():
            return {}

        result = {}
        for file in figi_dir.glob(f"{figi}_*.json"):
            timeframe = file.stem.replace(f"{figi}_", "")
            entries = self._load(file)
            result[timeframe] = entries
        return result
