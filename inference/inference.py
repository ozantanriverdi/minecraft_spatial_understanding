from PIL import Image
from qwen_model import Qwen_Model
from prompt_builder import PromptBuilder
from os.path import join
from utils import *
import json

samples_dir = "/home/atuin/v100dd/v100dd12/minecraft_agent/Minecraft-Video-Agent/spatial_evaluation/samples/sample_set_demo"

entity_counts = [2, 3, 4]
tasks = ["absolute_distance", "relative_distance", "relative_direction"]
biome_count = 10
traj_count = 10
prediction_base_dir = create_predictions_directory()



prompt_builder = PromptBuilder(samples_dir=samples_dir)
model = Qwen_Model()
sys_prompt = prompt_builder.sys_prompt()
for task in tasks:
    task_predictions = {}
    for entity_count in entity_counts:
        entity_key = f"trajectories_with_{entity_count}_entities"
        rgb_images_dir = Path(samples_dir) / entity_key / "rgb_frames"
        for biome in range(biome_count):
            for trajectory in range(traj_count):
                print(f"[{task}] Processing {entity_key} biome {biome}, trajectory {trajectory}")
                trajectory_key = f"{biome}_{trajectory}"
                user_prompt, sampled_entities = prompt_builder.user_prompt(task=task, entity_count=entity_count, biome=biome, trajectory=trajectory)

                with open(rgb_images_dir / f"{biome}_{trajectory}_0.jpg", "rb") as img:
                    rgb_frame = Image.open(img).convert("RGB")
                
                raw_output = model.forward(prompt=user_prompt, image=rgb_frame, system_prompt=sys_prompt)

                task_predictions.setdefault(entity_key, {})[trajectory_key] = {
                "raw_output": raw_output,
                "sampled_entities": sampled_entities
                }
    with open(join(prediction_base_dir, f"{task}_results.json"), "w") as result_f:
        json.dump(task_predictions, result_f, indent=2)

