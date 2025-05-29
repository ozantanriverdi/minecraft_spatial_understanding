from qwen_model import Qwen_Model
from PIL import Image

system_prompt = "You are an expert assistant evaluating spatial reasoning in Minecraft. Always respond in JSON format. [...]"
image = Image.open("seed_40.jpg").convert("RGB")

model = Qwen_Model()
output = model.forward(
    prompt="What is the distance from the agent to the red block?",
    image=image,
    system_prompt=system_prompt
)