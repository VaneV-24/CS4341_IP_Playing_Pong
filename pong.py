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
    def __init__(self, two_player=False):
        self.two_player = two_player

    def initial_state(self):
        ball_x = 0.5
        ball_y = 0.5
        velocity_x = 0.03
        velocity_y = 0.01
        # Right Paddle
        paddle_y = 0.5 - PADDLE_HEIGHT / 2

        if self.two_player:
            #Left Paddle
            paddle_x = 0.5 - PADDLE_HEIGHT / 2
            return ball_x, ball_y, velocity_x, velocity_y, paddle_y, paddle_x
        else:
            return ball_x, ball_y, velocity_x, velocity_y, paddle_y

    def step(self, state, action_y, action_x=None):
        if self.two_player:
            ball_x, ball_y, vx, vy, paddle_y, paddle_x = state
        else:
            ball_x, ball_y, vx, vy, paddle_y = state

        # Move paddle
        paddle_y += ACTION_EFFECTS[action_y]

        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - PADDLE_HEIGHT:
            paddle_y = 1 - PADDLE_HEIGHT

        if self.two_player:
            paddle_x += ACTION_EFFECTS[action_x]

            if paddle_x < 0:
                paddle_x = 0
            if paddle_x > 1 - PADDLE_HEIGHT:
                paddle_x = 1 - PADDLE_HEIGHT

        # Move ball
        ball_x += vx
        ball_y += vy

        reward_y = 0
        reward_x = 0
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

        # Left wall OR left paddle
        if self.two_player:
            if ball_x <= 0:
                if paddle_x <= ball_y <= paddle_x + PADDLE_HEIGHT:
                    ball_x = -ball_x
                    vx = -vx

                    # Add randomness
                    vx += random.uniform(-U_RANGE, U_RANGE)
                    vy += random.uniform(-V_RANGE, V_RANGE)

                    # Ensure minimum x velocity
                    if abs(vx) < MIN_VX:
                        vx = MIN_VX if vx > 0 else -MIN_VX

                    reward_x = +1
                else:
                    reward_x = -1
                    reward_y = +1
                    done = True
        else:
            if ball_x < 0:
                ball_x = -ball_x
                vx = -vx

        # Right Paddle collision
        if ball_x >= PADDLE_X:
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

                reward_y = +1

            else:
                # Missed paddle
                reward_y = -1
                reward_x = +1
                done = True
                ball_x = 1.1

        if self.two_player:
            return (ball_x, ball_y, vx, vy, paddle_y, paddle_x), reward_y, reward_x, done
        else:
            return (ball_x, ball_y, vx, vy, paddle_y), reward_y, done

    def discretize(self, state):

        if self.two_player:
            ball_x, ball_y, vx, vy, paddle_y, paddle_x = state

            # Terminal state
            if ball_x > 1:
                return 124416

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

            paddle_y_disc = int(math.floor(
                GRID_SIZE * paddle_y / (1 - PADDLE_HEIGHT)))

            if paddle_y_disc >= GRID_SIZE:
                paddle_y_disc = GRID_SIZE - 1

            paddle_x_disc = int(math.floor(
                GRID_SIZE * paddle_y / (1 - PADDLE_HEIGHT)))

            if paddle_x_disc >= GRID_SIZE:
                paddle_x_disc = GRID_SIZE - 1

            state_index = (
                    bx * (GRID_SIZE * 2 * 3 * GRID_SIZE * GRID_SIZE) +
                    by * (2 * 3 * GRID_SIZE * GRID_SIZE) +
                    (vx_disc == 1) * (3 * GRID_SIZE * GRID_SIZE) +
                    (vy_disc + 1) * (GRID_SIZE * GRID_SIZE) +
                    paddle_x_disc * GRID_SIZE +
                    paddle_y_disc)

            return state_index

        else:
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

            # Paddle y
            paddle_y_disc = int(math.floor(
                GRID_SIZE * paddle_y / (1 - PADDLE_HEIGHT)))

            if paddle_y_disc >= GRID_SIZE:
                paddle_y_disc = GRID_SIZE - 1

            # Combine into one to return
            state_index = (
                bx * (GRID_SIZE * 2 * 3 * GRID_SIZE) +
                by * (2 * 3 * GRID_SIZE) +
                (vx_disc == 1) * (3 * GRID_SIZE) +
                (vy_disc + 1) * GRID_SIZE +
                paddle_y_disc)

            return state_index