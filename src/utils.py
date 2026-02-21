"""
Utils for the project.
"""
import os
import json
from pathlib import Path
from datetime import datetime

def now_tag():
    """
    Returns a string with the current date and time in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    """
    Ensures that the directory at path p exists. If it doesn't exist, it is created.
    """
    Path(p).mkdir(parents=True, exist_ok=True)

def load_json(path):
    """
    Loads a JSON file from the given path and returns the corresponding Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    """
    Saves a Python object as a JSON file at the given path. The JSON file is formatted with 
    indentation for readability and does not escape non-ASCII characters.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def set_env_defaults():
    """
    Sets default environment variables for the project. This includes disabling parallelism in 
    tokenizers
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # cache local (evita que baje todo cada vez)
    os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
    ensure_dir(os.environ["HF_HOME"])
