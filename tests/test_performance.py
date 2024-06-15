import os
import sys
import pytest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import QLearningAgent, load_model, save_model, get_feedback

@pytest.fixture
def agent():
    return QLearningAgent()

def test_agent_initialization_time(agent):
    start_time = time.time()
    agent = QLearningAgent()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Initialization Time: {duration} seconds")

def test_action_selection_time(agent):
    available_actions = ["board", "candy", "delta"]
    state = (0, 0, 0, 0, 0)
    start_time = time.time()
    action = agent.choose_action(state, available_actions)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Action Selection Time: {duration} seconds")

def test_q_table_update_time(agent):
    available_actions = ["board", "candy", "delta"]
    state = (0, 0, 0, 0, 0)
    action = agent.choose_action(state, available_actions)
    start_time = time.time()
    agent.update_q_table(state, action, 1, (1, 1, 1, 1, 1), available_actions)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Q-table Update Time: {duration} seconds")