import pytest
from wordle_rl import QLearningAgent, load_model, get_feedback

@pytest.fixture
def agent():
    return load_model('q_learning_model.pkl')

def test_load_model():
    agent = load_model('q_learning_model.pkl')
    assert isinstance(agent, QLearningAgent), "Loaded agent should be an instance of QLearningAgent"

def test_get_feedback():
    target_word = "apple"
    assert get_feedback("april", target_word) == [2, 0, 1, 0, 0], "Feedback for 'april' should be [2, 0, 1, 0, 0]"
    assert get_feedback("apply", target_word) == [2, 0, 0, 1, 0], "Feedback for 'apply' should be [2, 0, 0, 1, 0]"
    assert get_feedback("orange", target_word) == [0, 1, 0, 0, 0], "Feedback for 'orange' should be [0, 1, 0, 0, 0]"

def test_qlearning_agent_initialization():
    agent = QLearningAgent()
    assert len(agent.q_table) == 0, "Q-table should be empty upon initialization"
    assert agent.learning_rate == 0.1, "Default learning rate should be 0.1"
    assert agent.discount_factor == 0.9, "Default discount factor should be 0.9"
    assert agent.epsilon == 0.3, "Default epsilon should be 0.3"

def test_qlearning_agent_choose_action(agent):
    state = (0, 0, 0, 0, 0)
    available_actions = ["apple", "orange", "banana"]
    action = agent.choose_action(state, available_actions)
    assert action in available_actions, f"Chosen action '{action}' should be in available actions"

def test_qlearning_agent_update_q_table(agent):
    state = (0, 0, 0, 0, 0)
    action = "apple"
    reward = 2
    next_state = (2, 0, 0, 0, 0)
    available_actions = ["apple", "orange", "banana"]
    agent.update_q_table(state, action, reward, next_state, available_actions)
    assert agent.q_table[state][action] != 0, "Q-value for state-action pair should be updated"

if __name__ == "__main__":
    pytest.main()
