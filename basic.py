import os
from pathlib import Path

from dotenv import load_dotenv

from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool, Pi3Tool


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

TOOL_SERVERS = {
    "depth": os.getenv("TOOL_SERVER_DEPTH", "http://0.0.0.0:20019"),
    "segmentation": os.getenv("TOOL_SERVER_SEGMENTATION", "http://0.0.0.0:20020"),
    "pi3": os.getenv("TOOL_SERVER_PI3", "http://127.0.0.1:20030"),
}

# Create model and tools
# model = GPTModel(model_name="gpt-4o-mini")
model = GPTModel(model_name="gemini-2.5-pro")
tools = [
    DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),    # Depth estimation
    SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["segmentation"]),  # Image segmentation
    Pi3Tool(use_mock=False, server_url=TOOL_SERVERS["pi3"], mode="inference"),
]

# Create agent
agent = SPAgent(model=model, tools=tools)

# Change to Qwen3-VL-4B-Instruct
# from spagent.models import QwenVLLMModel
# agent.set_model(QwenVLLMModel(model_name="Qwen/Qwen3-VL-4B-Instruct"), )

# Solve problem
result = agent.solve_problem("image.jpg", "Analyze the depth relationships and main objects in this image")
print(result['answer'])
