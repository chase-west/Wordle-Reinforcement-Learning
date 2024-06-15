# Wordle Reinforcement Learning
This project implements a reinforcement learning agent that learns to play the word guessing game Wordle. The agent uses Q-learning to optimize its guesses and improve over time.

# Requirements
To install the required packages, run:

- pip install -r requirements.txt

Contents of requirements.txt:

- wonderwords
- pyspellchecker
- pytest 
# Q-Learning Agent
The Q-learning agent is implemented in model.py. Key functionalities include:

- Initialization: Sets up Q-learning parameters and the Q-table.
- Choosing Actions: Uses an epsilon-greedy strategy to choose actions (guesses).
- Updating Q-Table: Updates the Q-table based on the feedback from the environment (game state).
  
# Usage
- Train the Agent: Run main.py to train the agent.
- Save the Model: The trained model is saved to q_learning_model.pkl.
- Load the Model: The model can be loaded for further testing or training using the load_model function.
# Example Output
During training, the agent's guesses and the number of correct guesses are printed:

Starting iteration 1/1000
Starting iteration 51/1000
...
Congratulations! The agent guessed the word 'apple' correctly in 3 attempts.
The agent guessed 123 words correctly!

# License
This project is licensed under the MIT License - see the LICENSE file for details.
