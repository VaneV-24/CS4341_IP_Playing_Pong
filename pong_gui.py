import pygame
import numpy as np
from pong import Pong

Q = np.load("q_table.npy")

WIDTH, HEIGHT = 600, 600
PADDLE_WIDTH = 10
BALL_SIZE = 8

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock()

pong = Pong()

def draw(state):
    screen.fill((0, 0, 0))

    ball_x, ball_y, vx, vy, paddle_y = state

    ball_px = int(ball_x * WIDTH)
    ball_py = int(ball_y * HEIGHT)

    paddle_py = int(paddle_y * HEIGHT)
    paddle_h = int(0.2 * HEIGHT)

    pygame.draw.circle(screen, (255, 255, 255),
                       (ball_px, ball_py), BALL_SIZE)

    pygame.draw.rect(screen, (255, 255, 255),
                     (WIDTH - PADDLE_WIDTH, paddle_py,
                      PADDLE_WIDTH, paddle_h))

    pygame.draw.rect(screen, (255, 255, 255),
                     (0, 1,
                      PADDLE_WIDTH, HEIGHT))

    pygame.display.flip()

def main():
    running = True

    state_cont = pong.initial_state()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state_disc = pong.discretize(state_cont)
        action = np.argmax(Q[state_disc])

        next_state, reward, done = pong.step(state_cont, action)
        state_cont = next_state

        draw(state_cont)

        if done:
            pygame.time.delay(500)
            state_cont = pong.initial_state()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()