from pathlib import Path
import json
from filelock import FileLock


def save_training_status(job_id: str, status: str, filename: str, filepath: Path = Path("training_status.json")):
    lock = FileLock(str(filepath) + ".lock")
    with lock:
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[job_id] = {
            "status": status,
            "filename": filename
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)


def load_training_status(filepath: Path = Path("training_status.json")) -> dict:
    lock = FileLock(str(filepath) + ".lock")
    with lock:
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return {}
