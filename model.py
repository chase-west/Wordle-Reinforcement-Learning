import pickle
from wonderwords import RandomWord
from spellchecker import SpellChecker
import random

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state, available_actions):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            return self.greedy_action(state, available_actions)

    def greedy_action(self, state, available_actions):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in available_actions}
        state_actions = self.q_table[state]
        return max(state_actions, key=state_actions.get)

    def update_q_table(self, state, action, reward, next_state, available_actions):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in available_actions}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        # Initialize next_state in Q-table if it doesn't exist
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in available_actions}

        max_next_q_value = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - self.q_table[state][action]
        )
        
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            agent = QLearningAgent()
            agent.q_table = data.get('q_table', {})  # Use default empty dictionary if 'q_table' key is missing
            agent.learning_rate = data.get('learning_rate', 0.1)
            agent.discount_factor = data.get('discount_factor', 0.9)
            agent.epsilon = data.get('epsilon', 0.1)
            return agent
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing key in the model file - {e}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def save_model(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'q_table': agent.q_table,
            'learning_rate': agent.learning_rate,
            'discount_factor': agent.discount_factor,
            'epsilon': agent.epsilon
        }, f)

correctGuess = 0
def run_episode(agent, available_actions, r):
    spell = SpellChecker()
    randWord = r.word(word_min_length=5, word_max_length=5)
    max_attempts = 6
    user_attempts = 0
    global correctGuess
    state = "start"

    while user_attempts < max_attempts:
        action = agent.choose_action(state, available_actions)
        
        new_state = action  # Update based on action and current state
        reward = 1 if action == randWord else -0.1 
            
        agent.update_q_table(state, action, reward, new_state, available_actions)
        state = new_state

        if action == randWord:
            correctGuess += 1
            print("Congratulations! The agent guessed the word correctly.")
            break
        else:
            print(f"The agent guessed '{action}' which is not correct.")
            user_attempts += 1

    if user_attempts == max_attempts:
        print(f"The agent did not guess the word within the maximum number of attempts. The word was {randWord}")

def main(num_episodes=100):
    agent = load_model('q_learning_model.pkl')
    if agent is None:
        agent = QLearningAgent()
        save_model(agent, 'q_learning_model.pkl')

    r = RandomWord()
    available_actions = [r.word(word_min_length=5, word_max_length=5) for _ in range(100)]

    for episode in range(num_episodes):
        run_episode(agent, available_actions, r)
        print(f"Episode {episode + 1}/{num_episodes} completed.")
    print(f"The model guessed {correctGuess} words correctly!")
    save_model(agent, 'q_learning_model.pkl')

if __name__ == "__main__":
    main(num_episodes=1000)  # Adjust the number of episodes here