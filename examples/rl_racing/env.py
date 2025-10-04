# Top-down racing environment for reinforcement learning.
# Uses pygame for rendering and follows Gymnasium API.

import math
from typing import List, Tuple, Optional

import numpy as np
import pygame
from gymnasium import Env, spaces


class RacingEnv(Env):
    """Simple top-down car racing environment."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        track: Optional[List[Tuple[float, float]]] = None,
        obstacles: Optional[List[pygame.Rect]] = None,
        num_sensors: int = 5,
        discrete: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.track = track or [
            (50, 300),
            (150, 300),
            (250, 250),
            (350, 200),
            (450, 200),
            (550, 250),
            (650, 300),
        ]
        self.obstacles = obstacles or [pygame.Rect(300, 270, 40, 40)]
        self.num_sensors = num_sensors
        self.discrete = discrete
        self.render_mode = render_mode
        self.dt = 0.1
        self.car_size = (20, 10)
        self.max_speed = 100.0
        self.sensor_range = 100.0

        if discrete:
            self.action_space = spaces.Discrete(4)  # fwd/back/left/right
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_low = np.array([0.0] * num_sensors + [0.0, -math.pi] + [-np.inf, -np.inf])
        obs_high = np.array([self.sensor_range] * num_sensors + [self.max_speed, math.pi] + [np.inf, np.inf])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._init_sim()
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((700, 400))
            pygame.display.set_caption("RacingEnv")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

    def _init_sim(self) -> None:
        self.pos = np.array(self.track[0], dtype=np.float32)
        self.vel = 0.0
        self.angle = 0.0
        self.wp_index = 1
        self.ghosts: List[Tuple[float, float]] = []

    def _update_physics(self, throttle: float, steering: float) -> None:
        self.vel += throttle * self.dt * 50
        self.vel = np.clip(self.vel, 0.0, self.max_speed)
        self.angle += steering * self.dt * 2
        dx = self.vel * math.cos(self.angle) * self.dt
        dy = self.vel * math.sin(self.angle) * self.dt
        self.pos += np.array([dx, dy], dtype=np.float32)

    def _check_collision(self) -> bool:
        car_rect = pygame.Rect(0, 0, *self.car_size)
        car_rect.center = self.pos
        for obs in self.obstacles:
            if car_rect.colliderect(obs):
                return True
        return False

    def _sensor_readings(self) -> List[float]:
        readings = []
        car_rect = pygame.Rect(0, 0, *self.car_size)
        car_rect.center = self.pos
        angles = np.linspace(-math.pi / 2, math.pi / 2, self.num_sensors)
        for ang in angles:
            total_ang = self.angle + ang
            for r in np.linspace(0, self.sensor_range, int(self.sensor_range / 5)):
                x = self.pos[0] + r * math.cos(total_ang)
                y = self.pos[1] + r * math.sin(total_ang)
                point = pygame.Rect(x, y, 2, 2)
                collision = any(point.colliderect(obs) for obs in self.obstacles)
                if collision:
                    readings.append(r)
                    break
            else:
                readings.append(self.sensor_range)
        return readings

    def _progress_reward(self, prev_dist: float, new_dist: float) -> float:
        return prev_dist - new_dist

    def step(self, action):
        if self.discrete:
            throttle = 1.0 if action == 0 else -1.0 if action == 1 else 0.0
            steering = 1.0 if action == 2 else -1.0 if action == 3 else 0.0
        else:
            throttle, steering = action
        prev_wp = np.array(self.track[self.wp_index - 1])
        target = np.array(self.track[self.wp_index])
        prev_dist = np.linalg.norm(target - self.pos)
        self._update_physics(throttle, steering)
        collision = self._check_collision()
        new_dist = np.linalg.norm(target - self.pos)
        reward = self._progress_reward(prev_dist, new_dist)
        terminated = False
        if new_dist < 15.0:
            reward += 1.0
            self.wp_index += 1
            if self.wp_index >= len(self.track):
                reward += 100.0
                terminated = True
        if collision:
            reward -= 5.0
            terminated = True
            self.ghosts.append(tuple(self.pos))
        obs = np.array(self._sensor_readings() + [self.vel, self.angle] + list(target - self.pos), dtype=np.float32)
        info = {}
        if terminated:
            info["is_success"] = self.wp_index >= len(self.track)
        return obs, reward, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_sim()
        obs = np.array(self._sensor_readings() + [self.vel, self.angle] + list(np.array(self.track[1]) - self.pos), dtype=np.float32)
        return obs, {}

    def render(self):
        if self.screen is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill((255, 255, 255))
        # Draw track
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.track, 3)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), obs)
        # Draw ghosts
        for g in self.ghosts:
            pygame.draw.circle(self.screen, (0, 0, 255, 100), (int(g[0]), int(g[1])), 5)
        # Draw car
        car_rect = pygame.Rect(0, 0, *self.car_size)
        car_rect.center = self.pos
        rotated = pygame.transform.rotate(pygame.Surface(self.car_size), -math.degrees(self.angle))
        rotated.fill((0, 255, 0))
        rect = rotated.get_rect(center=car_rect.center)
        self.screen.blit(rotated, rect.topleft)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
