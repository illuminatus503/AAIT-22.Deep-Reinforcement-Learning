import os
import glob
from typing import SupportsFloat

import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import torch as T
import numpy as np

from .agent import Agent


class DiscreteGamePlatform:
    def __load_agent(self, buffer_size, load_checkpoint):
        self._agent = Agent(
            lr=5e-4,
            gamma=0.99,
            epsilon=1.0,
            eps_min=1e-2,
            input_dims=self._input_dims,
            n_actions=self._n_actions,
            mem_size=buffer_size,  # Reducido, de 1.000.000 de ejemplos
            batch_size=64,
            replace=100,
            device=self._device,
            chkpt_dir=self._checkpoint_dir,
        )

        if load_checkpoint:
            self._agent.load_model()

    def __init__(self, env_id, buffer_size=50, load_checkpoint=False, **kwargs):
        # Params.
        self._device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self._checkpoint_dir = os.path.join("tmp", env_id)
        self._video_dir = os.path.join(self._checkpoint_dir, "videos")
        self._video_path = os.path.join(self._video_dir, f"{env_id}_render.mp4")

        # Load main env.
        self._env_id = env_id
        self._env_kwargs = kwargs
        self._env = gym.make(env_id, **kwargs)

        # Create agent.
        self._action_space = self._env.action_space
        if isinstance(self._action_space, Discrete):
            self._n_actions = int(self._action_space.n)
        else:
            raise ValueError("Entorno incompatible! Espacio de acciones continuo!")

        self._obs_space = self._env.observation_space
        self._input_dims = self._obs_space.shape

        print(f"Espacio de observaciones: {self._input_dims}")
        print(f"Espacio de acciones: {self._n_actions}")
        print(f"Tamaño del buffer de reproducción: {buffer_size} muestras S R A S' T")

        self.__load_agent(buffer_size, load_checkpoint)

    @staticmethod
    def __unwrap(state):
        if isinstance(state, tuple):
            return state[0]
        return state

    @staticmethod
    def __reset_env(env):
        return DiscreteGamePlatform.__unwrap(env.reset())

    @staticmethod
    def __get_next_state(env, action) -> tuple[None, SupportsFloat, int, dict]:
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = DiscreteGamePlatform.__unwrap(next_state)
        done = int(terminated and truncated)
        return (next_state, reward, done, info)

    def __run_training_step(self) -> float:
        done = False
        score = 0.0

        # Iniciamos el entorno en el estado inicial.
        state = DiscreteGamePlatform.__reset_env(self._env)
        while not done:
            action = self._agent.choose_action(state)
            next_state, reward, done, _ = DiscreteGamePlatform.__get_next_state(
                self._env, action
            )

            # Guardaríamos una observación y aprenderíamos.
            self._agent.store_transition(state, action, reward, next_state, done)
            self._agent.learn()

            # Avanzaríamos al siguiente momento
            state = next_state
            score += float(reward)

        return score

    def train_agent(
        self, num_games: int, game_sample: int = 100, save_period: int = 10
    ):
        score_stack = []
        prev_avg_score = -float("inf")

        for i in range(1, num_games + 1):
            score = self.__run_training_step()

            # Guardamos los resultados parciales
            score_stack.append(score)
            avg_score = np.mean(score_stack[-game_sample:])

            print(
                f"Episode T={i:4d} / {num_games} | "
                f"Score : {score:1.2f} | "
                f"Agent eps. : {self._agent.eps:1.3f} | "
                f"Avg. score, last {game_sample:3d} games : {avg_score:1.3f}"
            )

            if i % save_period == 0:
                if prev_avg_score < avg_score:
                    print("Saving model...")
                    self._agent.save_model()

                    prev_avg_score = avg_score
                else:
                    print(f"Current score: {avg_score:1.3f} < {prev_avg_score:1.3f}")
                    print("Restoring prev. model...")
                    self._agent.load_model()
                    del score_stack[-save_period:]

    def __create_video_env(self):
        self._rgb_env = gym.make(
            self._env_id, render_mode="rgb_array", **self._env_kwargs
        )

        self._video_recorder = VideoRecorder(
            self._rgb_env, self._video_path, enabled=True
        )

    def __render_game(self) -> float:
        done = False
        score = 0.0

        # Iniciamos el entorno en el estado inicial.
        state = DiscreteGamePlatform.__reset_env(self._rgb_env)
        while not done:
            self._rgb_env.unwrapped.render()
            self._video_recorder.capture_frame()

            action = self._agent.choose_action(state)
            next_state, reward, done, _ = DiscreteGamePlatform.__get_next_state(
                self._rgb_env, action
            )

            # Avanzaríamos al siguiente momento
            state = next_state
            score += float(reward)

        return score

    def __remove_videos(self):
        video_dir_content = glob.glob(os.path.join(self._video_dir, "*.mp4"))
        for video in video_dir_content:
            try:
                os.remove(video)
            except:
                print(f"Error while deleting file: {video}")

    def record_play(self):
        # Eliminar los vídeos antiguos
        try:
            os.makedirs(self._video_dir, exist_ok=False)
        except OSError:
            print("Directory already exists!")
            self.__remove_videos()

        # Render game
        self.__create_video_env()
        score = self.__render_game()
        self._rgb_env.close()
        self._video_recorder.close()

        return score

    def close(self):
        self._env.close()
