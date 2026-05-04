import pygame
import numpy as np
from pong import Pong
import sys

MODE = "single"

if len(sys.argv) > 1:
    MODE = sys.argv[1]

if MODE == "two":
    Q_y = np.load("q_y_table.npy")
    Q_x = np.load("q_x_table.npy")
else:
    Q = np.load("q_table.npy")

WIDTH, HEIGHT = 600, 600
PADDLE_WIDTH = 10
BALL_SIZE = 8

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

pong = Pong(two_player=(MODE == "two"))

def draw(state):
    screen.fill((0, 0, 0))

    if MODE == "two":
        ball_x, ball_y, vx, vy, paddle_y, paddle_x = state
    else:
        ball_x, ball_y, vx, vy, paddle_y = state

    ball_px = int(ball_x * WIDTH)
    ball_py = int(ball_y * HEIGHT)

    paddle_py = int(paddle_y * HEIGHT)

    if MODE == "two":
        paddle_px = int(paddle_x * HEIGHT)

    paddle_h = int(0.2 * HEIGHT)

    pygame.draw.circle(screen, (255, 255, 255),
                       (ball_px, ball_py), BALL_SIZE)

    pygame.draw.rect(screen, (255, 255, 255),
                     (WIDTH - PADDLE_WIDTH, paddle_py,
                      PADDLE_WIDTH, paddle_h))

    if MODE == "two":
        pygame.draw.rect(screen, (255, 255, 255),
                         (0, paddle_px,
                          PADDLE_WIDTH, paddle_h))
    else:
        pygame.draw.rect(screen, (255, 255, 255),
                         (0, 1,
                          PADDLE_WIDTH, HEIGHT))

    pygame.display.flip()

def main():
    running = True

    state_cont = pong.initial_state()

    game_number = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.set_caption(f"Pong ({MODE}-player) | Game: {game_number}")

        if MODE == "two":
            state_disc = pong.discretize(state_cont)
            action_y = np.argmax(Q_y[state_disc])
            action_x = np.argmax(Q_x[state_disc])

            next_state, reward_y, reward_x, done = pong.step(state_cont, action_y, action_x)
            state_cont = next_state

        else:
            state_disc = pong.discretize(state_cont)
            action = np.argmax(Q[state_disc])

            next_state, reward, done = pong.step(state_cont, action)
            state_cont = next_state

        draw(state_cont)

        if done:
            pygame.time.delay(500)
            game_number += 1
            state_cont = pong.initial_state()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()