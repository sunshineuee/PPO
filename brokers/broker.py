# broker.py

from tinkoff_api import TinkoffAPI
from okx_api import OKXAPI

class Broker:
    _brokers = {
        "TINKOFF": TinkoffAPI(),
        "OKX": OKXAPI(),
    }

    @classmethod
    def get_price(cls, broker_name, figi):
        broker_api = cls._brokers[broker_name]
        return broker_api.get_price(figi)

    @classmethod
    def buy(cls, broker_name, figi, amount):
        broker_api = cls._brokers[broker_name]
        return broker_api.buy(figi, amount)

    @classmethod
    def sell(cls, broker_name, figi, amount):
        broker_api = cls._brokers[broker_name]
        return broker_api.sell(figi, amount)

def get_broker(broker_name: str):
    """
    Возвращает объект брокера по имени (TINKOFF, OKX и т.д.)
    """
    return Broker._brokers.get(broker_name.upper())