import os
import sys
import pytest
import random
import pickle

# Adjust the path to include the parent directory of model.py and q_learning_model.pkl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import QLearningAgent, load_model, save_model, get_feedback



@pytest.fixture
def agent():
    model_path = os.path.join(os.path.dirname(__file__), '../q_learning_model.pkl')
    return load_model(model_path)

def test_load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../q_learning_model.pkl')
    agent = load_model(model_path)
    assert isinstance(agent, QLearningAgent), "Loaded agent should be an instance of QLearningAgent"

def test_save_load_roundtrip(tmp_path):
    # Create a temporary directory for testing
    tmp_file = tmp_path / "test_model.pkl"
    
    # Create an instance of QLearningAgent and save it
    agent = QLearningAgent()
    save_model(agent, tmp_file)
    
    # Load the saved model and assert its type
    loaded_agent = load_model(tmp_file)
    assert isinstance(loaded_agent, QLearningAgent), "Loaded agent should be an instance of QLearningAgent"

def test_get_feedback():
    target_word = "apple"
    assert get_feedback("april", target_word) == [2, 0, 1, 0, 0], "Feedback for 'april' should be [2, 0, 1, 0, 0]"
    assert get_feedback("apply", target_word) == [2, 0, 0, 1, 0], "Feedback for 'apply' should be [2, 0, 0, 1, 0]"
    assert get_feedback("orange", target_word) == [0, 1, 0, 0, 0], "Feedback for 'orange' should be [0, 1, 0, 0, 0]"

def test_q_learning_agent_methods():
    agent = QLearningAgent()
    # Test action selection methods (choose_action, greedy_action)
    available_actions = ["apple", "banana", "cherry"]
    state = (0, 0, 0, 0, 0)
    action = agent.choose_action(state, available_actions)
    assert action in available_actions, f"Action {action} should be in available actions {available_actions}"

    # Test update_q_table method
    agent.update_q_table(state, action, 1, (1, 1, 1, 1, 1), available_actions)
    assert state in agent.q_table, "State should be in Q-table after update"