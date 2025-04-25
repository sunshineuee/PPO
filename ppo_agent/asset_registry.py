import json
import os

class AssetRegistry:
    def __init__(self, file_path='asset_registry.json'):
        self.file_path = file_path
        self.registry = self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_registry(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def get_broker_and_figi(self, figi):
        if figi not in self.registry:
            # Автоматически добавляем новый FIGI с брокером TINKOFF
            self.registry[figi] = "TINKOFF"
            self._save_registry()
        return self.registry[figi], figi

    def set_broker(self, figi, broker):
        self.registry[figi] = broker
        self._save_registry()

# Использование:
asset_registry = AssetRegistry()
# asset_registry.py

ASSET_REGISTRY = {
    "BTC": ("OKX", "BTC-USDT"),
    "ETH": ("OKX", "ETH-USDT"),
    "SBER": ("TINKOFF", "BBG004730N88"),
    # ... добавляй активы сюда
}

def get_broker_and_figi(asset_name: str):
    return ASSET_REGISTRY.get(asset_name)