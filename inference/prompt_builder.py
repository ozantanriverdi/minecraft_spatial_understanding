import json
import random
from os.path import join

class PromptBuilder:
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        with open("prompt_template.json", "r") as template:
            self.prompt_template = json.load(template)

    def sys_prompt(self):
        return self.prompt_template.get("system")
    
    def user_prompt(self, task, entity_count, biome, trajectory):
        user_prompt_template = self.prompt_template.get(f"user_{task}")
        sampled_entities = self._sample_entities(task, entity_count, biome, trajectory)

        format_args = {
            "entity_1": sampled_entities[0],
            "entity_2": sampled_entities[1] if len(sampled_entities) > 1 else "",  # Safe default
            "distance_instructions": self.prompt_template.get("distance_instructions", ""),
            "direction_instructions": self.prompt_template.get("direction_instructions", ""),
            "world_knowledge_distance": self.prompt_template.get("world_knowledge_distance", ""),
            "world_knowledge_direction": self.prompt_template.get("world_knowledge_direction", ""),
            "direction_examples": self.prompt_template.get("direction_examples", "")
        }

        user_prompt = user_prompt_template.format(**format_args)
        return user_prompt, sampled_entities

    def _sample_entities(self, task, entity_count, biome, trajectory):
        traj_info_dir = join(self.samples_dir, f"trajectories_with_{entity_count}_entities/info")
        with open(join(traj_info_dir, f"info_step_{biome}_{trajectory}")) as f:
            info = json.load(f)
        entities = info.get("entities_spawned")
        if task == "absolute_distance":
            sampled_entities = random.sample(entities, 1)
        elif task in ["relative_distance", "relative_direction"]:
            sampled_entities = random.sample(entities, 2)
        return sampled_entities
