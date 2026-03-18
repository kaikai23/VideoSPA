from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, SegmentationTool

# Create model and tools
model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),    # Depth estimation
    SegmentationTool(use_mock=True)        # Image segmentation
]

# Create agent
agent = SPAgent(model=model, tools=tools)

# Solve problem
result = agent.solve_problem("image.jpg", "Analyze the depth relationships and main objects in this image")
print(result['answer'])
