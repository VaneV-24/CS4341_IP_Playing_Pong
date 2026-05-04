import numpy as np
import random
from pong import Pong
import sys

MODE = "single"

if len(sys.argv) > 1:
    MODE = sys.argv[1]

# Note: blank_y represents values that are associated with paddle y and
# blank_x represents values that are associated with paddle x

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

Q_2 = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))
N_2 = np.zeros((NUM_STATES_2_PLAYER, NUM_ACTIONS))

ACTIONS = [0, 1, 2]

def select_action(q_table, state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return np.argmax(q_table[state])

def update_q(q_table, n_table, state, action, reward, next_state):
    n_table[state][action] += 1
    alpha = C / (C + n_table[state][action])
    best_next = float(np.max(q_table[next_state]))
    q_table[state][action] += alpha * (reward + gamma *
                                 best_next - q_table[state][action])

def perspective(state, player):
    ball_x, ball_y, vx, vy, paddle_y, paddle_x = state

    if player == "x": #left paddle
        ball_x = 1 - ball_x
        vx = -vx
        paddle_y, paddle_x = paddle_x, paddle_y

    return ball_x, ball_y, vx, vy, paddle_y, paddle_x

def train(two_player=(MODE == "two")):
    pong = Pong(two_player=two_player)

    global epsilon

    for episode in range(episodes):

        # Start new episode
        state_cont = pong.initial_state()
        done = False
        bounces = 0

        while not done:
            if two_player:
                # LEFT player (y)
                state_y = pong.discretize(perspective(state_cont, "y"))
                action_y = select_action(Q_2, state_y)

                # RIGHT player (x)
                state_x = pong.discretize(perspective(state_cont, "x"))
                action_x = select_action(Q_2, state_x)

                next_state_cont, reward_y, reward_x, done = pong.step(state_cont,
                    action_y, action_x)

                next_state_y = pong.discretize(perspective(next_state_cont, "y"))
                next_state_x = pong.discretize(perspective(next_state_cont, "x"))

                update_q(Q_2, N_2, state_y, action_y, reward_y, next_state_y)
                update_q(Q_2, N_2, state_x, action_x, reward_x, next_state_x)

                if reward_y == 1 or reward_x == 1:
                    bounces += 1

            else:
                state = pong.discretize(state_cont)

                action = select_action(Q, state)

                next_state_cont, reward, done = pong.step(state_cont, action)

                next_state = pong.discretize(next_state_cont)

                update_q(Q, N, state, action, reward, next_state)

                if reward == 1:
                    bounces += 1

            state_cont = next_state_cont

        if pong.two_player:
            epsilon = max(0.05, epsilon * 0.99999)
        else:
            epsilon = max(0.01, epsilon * 0.99995)

        if episode % 10000 == 0:
            print(
                f"Episode {episode} | "
                f"Bounces: {bounces} | "
                f"Epsilon: {epsilon:.4f}"
            )

    if pong.two_player:
        np.save("q_2p_table.npy", Q_2)
    else:
        np.save("q_table.npy", Q)


def evaluate(two_player=(MODE == "two")):
    pong = Pong(two_player=two_player)

    total_bounces = 0

    for _ in range(eval_games):
        state_cont = pong.initial_state()
        done = False
        bounces = 0

        while not done:
            if two_player:
                state_y = pong.discretize(perspective(state_cont, "y"))
                action_y = np.argmax(Q_2[state_y])

                state_x = pong.discretize(perspective(state_cont, "x"))
                action_x = np.argmax(Q_2[state_x])

                next_state_cont, reward_y, reward_x, done = pong.step(state_cont,
                    action_y, action_x)

                if reward_y == 1 or reward_x == 1:
                    bounces += 1

            else:
                state = pong.discretize(state_cont)
                action = select_action(Q, state)

                next_state_cont, reward, done = pong.step(state_cont, action)

                if reward == 1:
                    bounces += 1

            state_cont = next_state_cont

        total_bounces += bounces

    avg = total_bounces / eval_games

    print(f"Average bounces over {eval_games} games: {avg}")


if __name__ == "__main__":

    if MODE == "two":
        train(two_player=True)
        evaluate(two_player=True)
    else:
        train()
        evaluate()
