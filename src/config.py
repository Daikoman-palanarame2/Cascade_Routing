import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "http://localhost:8000/v1")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-3-4b-it")
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "gemma-3-27b-it")

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.96"))

# Load calibrated threshold if available, otherwise default to 0.72
default_threshold = 0.72
config_dir = os.path.dirname(__file__)
threshold_json_path = os.path.join(config_dir, "calibration", "threshold.json")
if os.path.exists(threshold_json_path):
    import json
    try:
        with open(threshold_json_path, "r") as f:
            data = json.load(f)
            default_threshold = float(data.get("threshold", 0.72))
    except Exception:
        pass

ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", str(default_threshold)))
SELF_CONSISTENCY_N = int(os.getenv("SELF_CONSISTENCY_N", "3"))
SELF_REFINE_ROUNDS = int(os.getenv("SELF_REFINE_ROUNDS", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
SIMULATE_LOCAL = os.getenv("SIMULATE_LOCAL", "True").lower() in ("true", "1", "yes")
MOCK_LLM = os.getenv("MOCK_LLM", "False").lower() in ("true", "1", "yes")
