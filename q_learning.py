import numpy as np
import random
from pong import Pong

# Hyperparameters
NUM_STATES = 10369
NUM_STATES_2_PLAYER = 124417

NUM_ACTIONS = 3

# Learning rate constant
C = 100

# Discount factor
gamma = 0.9

# Exploration rate
epsilon = 0.1

episodes = 100000
eval_games = 1000

# Q-learning tables
Q = np.zeros((NUM_STATES, NUM_ACTIONS))
N = np.zeros((NUM_STATES, NUM_ACTIONS))

Q_y = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))
Q_left = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))

N_y = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))
N_left = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))

ACTIONS = [0, 1, 2]

def select_action(pong, state):
    if pong.two_player:
        if random.random() < epsilon:
            action_y = np.random.choice(ACTIONS)
            return random.choice(ACTIONS)
    else:
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
    if pong.two_player:
        return np.argmax(Q_2[state])
    else:
        return np.argmax(Q[state])

def update_q(pong, state, action, reward, next_state):

    if pong.two_player:
        N_2[state][action] += 1
        alpha = C / (C + N_2[state][action])
        best_next = np.max(Q_2[next_state])
        Q_2[state][action] += alpha * (reward + gamma *
                                     best_next - Q_2[state][action])
    else:
        N[state][action] += 1
        alpha = C / (C + N[state][action])
        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma *
                                       best_next - Q[state][action])

def train():
    pong = Pong(two_player=True)

    global epsilon

    for episode in range(episodes):

        # Start new episode
        state_cont = pong.initial_state()
        state_disc = pong.discretize(state_cont)

        done = False
        bounces = 0

        while not done:

            action_y = select_action(pong, state_disc)
            action_left = select_action(pong, state_disc)

            next_state_cont, reward, done = pong.step(state_cont, action_y,
                                                      action_left)

            next_state = pong.discretize(next_state_cont)

            update_q(pong, state_disc, action_y, reward, next_state)

            if reward == 1:
                bounces += 1

            state_cont = next_state_cont
            state_disc = next_state

        epsilon = max(0.01, epsilon * 0.99995)

        if episode % 10000 == 0:
            print(
                f"Episode {episode} | "
                f"Bounces: {bounces} | "
                f"Epsilon: {epsilon:.4f}"
            )
    if pong.two_player:
        np.save("q_table.npy", Q_2)
    else:
        np.save("q_table.npy", Q)


def evaluate():
    pong = Pong(two_player=True)

    total_bounces = 0

    for _ in range(eval_games):

        state_cont = pong.initial_state()
        state_disc = pong.discretize(state_cont)

        done = False
        bounces = 0

        while not done:

            action_y = np.argmax(Q_2[state_disc])
            action_left = np.argmax(Q_2[state_disc])

            next_state_cont, reward, done = pong.step(state_cont, action_y, action_left)

            state_cont = next_state_cont
            state_disc = pong.discretize(state_cont)

            if reward == 1:
                bounces += 1

        total_bounces += bounces

    avg = total_bounces / eval_games

    print(f"Average bounces over {eval_games} games: {avg}")


if __name__ == "__main__":
    train()
    evaluate()
