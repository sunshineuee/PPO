import logging
import uvicorn

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Запуск PPO Agent API...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
