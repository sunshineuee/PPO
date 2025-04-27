import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
from threading import Lock
import re
from functools import lru_cache
from contextlib import contextmanager

class FIGIValidationError(Exception):
    pass

class AssetRegistryError(Exception):
    pass

class AssetRegistry:
    FIGI_PATTERN = re.compile(r'^BBG[A-Z0-9]{9}$|^[A-Z0-9]{12}$')
    
    def __init__(self, file_path='asset_registry.json', max_cache_size=1000):
        self.file_path = Path(file_path)
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._ensure_registry_exists()
        self.cache_size = max_cache_size

    def _setup_logging(self):
        handler = logging.FileHandler('asset_registry.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _ensure_registry_exists(self):
        """Создает структуру реестра, если она не существует"""
        if not self.file_path.exists():
            self._save_registry({
                "assets": {},
                "metadata": {
                    "version": "2.0",
                    "created_at": datetime.utcnow().isoformat(),
                    "last_backup": None
                }
            })

    @contextmanager
    def _file_lock(self):
        """Контекстный менеджер для безопасной работы с файлом"""
        with self.lock:
            try:
                yield
            except Exception as e:
                self.logger.error(f"Error during file operation: {e}")
                raise AssetRegistryError(f"Registry operation failed: {e}")

    @lru_cache(maxsize=1000)
    def _validate_figi(self, figi: str) -> bool:
        """Проверяет формат FIGI"""
        if not isinstance(figi, str):
            return False
        return bool(self.FIGI_PATTERN.match(figi))

    def _load_registry(self) -> Dict:
        """Загружает реестр с обработкой ошибок"""
        with self._file_lock():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                # Валидация структуры данных
                if not isinstance(data, dict) or "assets" not in data:
                    raise ValueError("Invalid registry structure")
                return data
            except json.JSONDecodeError as e:
                self.logger.error(f"Registry file corrupted: {e}")
                # Создаем бэкап проблемного файла
                if self.file_path.exists():
                    backup_path = self.file_path.with_suffix('.json.bak')
                    self.file_path.rename(backup_path)
                return {"assets": {}, "metadata": {}}

    def _save_registry(self, data: Dict) -> None:
        """Сохраняет реестр с созданием бэкапа"""
        with self._file_lock():
            # Создаем бэкап перед сохранением
            if self.file_path.exists():
                backup_path = self.file_path.with_suffix(f'.json.bak{datetime.now().strftime("%Y%m%d%H%M%S")}')
                self.file_path.rename(backup_path)
            
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            data["metadata"]["last_backup"] = datetime.utcnow().isoformat()

    def register_figi(self, figi: str, broker: str, additional_info: Optional[Dict] = None) -> None:
        """Регистрирует новый FIGI с расширенной валидацией"""
        if not self._validate_figi(figi):
            raise FIGIValidationError(f"Invalid FIGI format: {figi}")

        registry = self._load_registry()
        now = datetime.utcnow().isoformat()

        try:
            if figi not in registry["assets"]:
                registry["assets"][figi] = {
                    "broker": broker,
                    "first_seen": now,
                    "last_updated": now,
                    "status": "active",
                    "update_history": [],
                    "additional_info": additional_info or {}
                }
            else:
                asset = registry["assets"][figi]
                # Сохраняем историю изменений
                asset["update_history"].append({
                    "timestamp": now,
                    "broker": asset["broker"],
                    "status": asset["status"]
                })
                asset["broker"] = broker
                asset["last_updated"] = now
                if additional_info:
                    asset["additional_info"].update(additional_info)

            self._save_registry(registry)
            self.logger.info(f"Successfully registered/updated FIGI: {figi}")

        except Exception as e:
            self.logger.error(f"Error registering FIGI {figi}: {e}")
            raise AssetRegistryError(f"Failed to register FIGI: {e}")

    def get_inactive_assets(self, days_threshold: int = 30) -> list:
        """Находит неактивные активы"""
        registry = self._load_registry()
        inactive = []
        threshold = datetime.utcnow() - pd.Timedelta(days=days_threshold)
        
        for figi, data in registry["assets"].items():
            last_updated = datetime.fromisoformat(data["last_updated"])
            if last_updated < threshold:
                inactive.append(figi)
        
        return inactive

    def cleanup_old_records(self, days_threshold: int = 90):
        """Очищает старые записи"""
        registry = self._load_registry()
        threshold = datetime.utcnow() - pd.Timedelta(days=days_threshold)
        
        for figi in list(registry["assets"].keys()):
            data = registry["assets"][figi]
            last_updated = datetime.fromisoformat(data["last_updated"])
            if last_updated < threshold and data["status"] != "active":
                del registry["assets"][figi]
                self.logger.info(f"Removed old record: {figi}")
        
        self._save_registry(registry)