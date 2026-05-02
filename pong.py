import random
import math

PADDLE_HEIGHT = 0.2
PADDLE_X = 1.0

GRID_SIZE = 12

PADDLE_MOVE = 0.04

MIN_VX = 0.03

ACTIONS = [0, 1, 2]

ACTION_EFFECTS = {
    0: 0,
    1: PADDLE_MOVE,
    2: -PADDLE_MOVE
}

# Random bounce noise
U_RANGE = 0.015
V_RANGE = 0.03

class Pong:
    def initial_state(self):
        ball_x = 0.5
        ball_y = 0.5
        velocity_x = 0.03
        velocity_y = 0.01
        paddle_y = 0.5 - PADDLE_HEIGHT / 2

        return ball_x, ball_y, velocity_x, velocity_y, paddle_y

    def step(self, state, action):
        ball_x, ball_y, vx, vy, paddle_y = state

        # Move paddle
        paddle_y += ACTION_EFFECTS[action]

        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - PADDLE_HEIGHT:
            paddle_y = 1 - PADDLE_HEIGHT

        # Move ball
        ball_x += vx
        ball_y += vy

        reward = 0
        done = False

        # Wall collisions

        # Top wall
        if ball_y < 0:
            ball_y = -ball_y
            vy = -vy

        # Bottom wall
        if ball_y > 1:
            ball_y = 2 - ball_y
            vy = -vy

        # Left wall
        if ball_x < 0:
            ball_x = -ball_x
            vx = -vx

        # Paddle collision
        if ball_x >= PADDLE_X and vx > 0:
            if paddle_y <= ball_y <= paddle_y + PADDLE_HEIGHT:
                # Bounce
                ball_x = 2 * PADDLE_X - ball_x
                vx = -vx

                # Add randomness
                vx += random.uniform(-U_RANGE, U_RANGE)
                vy += random.uniform(-V_RANGE, V_RANGE)

                # Ensure minimum x velocity
                if abs(vx) < MIN_VX:
                    vx = MIN_VX if vx > 0 else -MIN_VX

                vx = max(min(vx, 1), -1)
                vy = max(min(vy, 1), -1)

                reward = +1

            else:
                # Missed paddle
                reward = -1
                done = True
                ball_x = 1.1

        return (ball_x, ball_y, vx, vy, paddle_y), reward, done

    def discretize(self, state):
        ball_x, ball_y, vx, vy, paddle_y = state

        # Terminal state
        if ball_x > 1:
            return 10368

        # Ball position
        bx = min(int(ball_x * GRID_SIZE), GRID_SIZE - 1)
        by = min(int(ball_y * GRID_SIZE), GRID_SIZE - 1)

        # Velocity X
        vx_disc = 1 if vx > 0 else -1

        # Velocity Y
        if abs(vy) < 0.015:
            vy_disc = 0
        elif vy > 0:
            vy_disc = 1
        else:
            vy_disc = -1

        # Paddle
        paddle_disc = int(math.floor(
            GRID_SIZE * paddle_y / (1 - PADDLE_HEIGHT)))

        if paddle_disc >= GRID_SIZE:
            paddle_disc = GRID_SIZE - 1

        # Combine into one to return
        state_index = (
            bx * (GRID_SIZE * 2 * 3 * GRID_SIZE) +
            by * (2 * 3 * GRID_SIZE) +
            (vx_disc == 1) * (3 * GRID_SIZE) +
            (vy_disc + 1) * GRID_SIZE +
            paddle_disc)

        return state_index