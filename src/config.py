import os
import yaml
from dataclasses import dataclass


@dataclass
class Config:
    def __init__(self, yaml_path: str = "config.yaml"):
        # Pfad-Sicherheit, falls aus Unterordnern aufgerufen
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(os.path.dirname(__file__), "..", yaml_path)

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Zuweisung der Dictionaries
        self.paths = cfg["paths"]
        self.data_split = cfg["data_split"]
        self.model = cfg["model"]
        self.training = cfg["training"]
