import os
from dataclasses import dataclass

import yaml


@dataclass
class Config:
    def __init__(self, yaml_path: str = "config.yaml"):
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(os.path.dirname(__file__), "..", yaml_path)

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.paths = cfg["paths"]
        self.data_split = cfg["data_split"]
        self.model = cfg["model"]
        self.training = cfg["training"]
