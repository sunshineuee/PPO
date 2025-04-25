import io
import asyncio
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
from multiprocessing import Process

from ppo_agent.agent import PPOAgent
from ppo_agent.data_loader import load_recent_candles, load_from_csv
from ppo_agent.status import save_training_status, load_training_status

from fastapi import HTTPException, Query
from ppo_agent.utils import TrainingJournal

# Укажи путь к журналу обучения

training_journal = TrainingJournal("ppo_agent/training_journal.json")
app = FastAPI(title="PPO Agent API")
agent = PPOAgent()


class InferenceRequest(BaseModel):
    asset: str


class InferenceResponse(BaseModel):
    predictions: Dict[str, Dict[str, float]]


@app.post("/predict", response_model=InferenceResponse)
async def predict_growth(request: InferenceRequest):
    result = {}
    for timeframe in agent.get_trained_timeframes(request.asset):
        try:
            df = load_recent_candles(request.asset, timeframe)
            signal, confidence = agent.predict(request.asset, timeframe, df)
            result[timeframe] = {"signal": signal, "confidence": confidence}
        except Exception as e:
            result[timeframe] = {"error": str(e)}
    return {"predictions": result}


@app.post("/train/csv")
async def train_from_csv(file: UploadFile = File(...)):
    try:
        job_id = str(uuid.uuid4())
        filename = file.filename
        contents = await file.read()
        csv_data = io.StringIO(contents.decode("utf-8"))
        df = pd.read_csv(csv_data)

        def background_train():
            try:
                save_training_status(job_id, "started", filename)
                df_loaded = load_from_csv(io.StringIO(contents.decode("utf-8")))
                grouped = df_loaded.groupby(["asset", "timeframe"])
                for (asset, timeframe), group in grouped:
                    features = group.drop(columns=["asset", "timeframe"])
                    agent.train(asset, timeframe, features)
                save_training_status(job_id, "finished", filename)
            except Exception as e:
                save_training_status(job_id, f"failed: {e}", filename)

        process = Process(target=background_train)
        process.start()

        return {"status": "training started", "job_id": job_id, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {e}")


@app.get("/train/status")
async def get_training_status():
    return load_training_status()




@app.on_event("startup")
async def continuous_online_training():
    async def train_loop():
        while True:
            for asset in agent.get_assets():
                for timeframe in agent.get_trained_timeframes(asset):
                    try:
                        df = load_recent_candles(asset, timeframe)
                        agent.train(asset, timeframe, df)
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"[TRAIN ERROR] {asset} {timeframe}: {e}")
            await asyncio.sleep(60)

    asyncio.create_task(train_loop())

@app.get("/training/intervals")
async def get_training_intervals(figi: str = Query(..., description="FIGI актива")):
    """
    Возвращает список интервалов, на которых обучалась модель для указанного FIGI.
    """
    result = {}
    for key, intervals in training_journal.data.items():
        if key.startswith(figi):
            _, timeframe = key.split("_", 1)
            result[timeframe] = intervals
    if not result:
        raise HTTPException(status_code=404, detail=f"Нет обученных интервалов для {figi}")
    return {"figi": figi, "trained_intervals": result}