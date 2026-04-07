import json
import os
from pathlib import Path

from dotenv import load_dotenv

from spagent import SPAgent
from spagent.models import QwenVLLMModel
from spagent.tools import DepthEstimationTool, SegmentationTool, Pi3Tool


def load_holispatial_sample_by_id(jsonl_path: Path, question_id: str) -> dict:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if str(sample.get("id")) == str(question_id):
                return sample
    raise KeyError(f"question_id not found: {question_id}")


def extract_question(sample: dict) -> str:
    for msg in sample.get("conversations", []):
        if msg.get("from") == "human":
            text = msg.get("value", "")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and ln.strip() != "<image>"]
            return "\n".join(lines)
    raise ValueError("No human question found in conversations")


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

CONFIG = {
    "data": {
        "root": "./HoliSpatial-QA-2M",
        "jsonl": "HoliSpatial-QA-2M.jsonl",
        "question_id": "81623",
        "num_images": 2,
    },
    "model": {
        "name": "Qwen/Qwen3-VL-4B-Instruct",
    },
    "tools": {
        "depth": {
            "use_mock": False,
            "server_url": os.getenv("TOOL_SERVER_DEPTH", "http://127.0.0.1:20019"),
        },
        "segmentation": {
            "use_mock": False,
            "server_url": os.getenv("TOOL_SERVER_SEGMENTATION", "http://127.0.0.1:20020"),
        },
        "pi3": {
            "use_mock": False,
            "server_url": os.getenv("TOOL_SERVER_PI3", "http://127.0.0.1:20030"),
            "mode": "inference",
        },
    },
}

# Load config
data_config = CONFIG["data"]
tool_config = CONFIG["tools"]
data_root = Path(data_config["root"])
jsonl_path = data_root / data_config["jsonl"]

# Load sample
sample = load_holispatial_sample_by_id(jsonl_path, data_config["question_id"])
question = extract_question(sample)

image_list = sample.get("image", [])
image_paths = [str((data_root / p).resolve()) for p in image_list[: data_config["num_images"]]]

model = QwenVLLMModel(model_name=CONFIG["model"]["name"])
tools = [
    DepthEstimationTool(**tool_config["depth"]),
    SegmentationTool(**tool_config["segmentation"]),
    Pi3Tool(**tool_config["pi3"]),
]

# Create agent
agent = SPAgent(model=model, tools=tools)

# Solve problem
result = agent.solve_problem(image_paths, question)
print(result['answer'])
