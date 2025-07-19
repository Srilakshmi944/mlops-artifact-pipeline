import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_config, train_model, load_data


import json
import pytest
from sklearn.linear_model import LogisticRegression


def test_config_loading():
    config = load_config("config/config.json")
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training_type():
    X, y = load_data()
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)

def test_model_training_accuracy():
    X, y = load_data()
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    acc = model.score(X, y)
    assert acc > 0.9

def test_dummy():
    assert 1 + 1 == 2
