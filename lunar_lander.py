from typing import SupportsFloat
import glob
import os
import sys

import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np

from src.agent import Agent


def load_env(env_id: str) -> gym.Env:
    return gym.make(id=env_id, continuous=False)


def load_agent(load_checkpoint: bool = False) -> Agent:
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=5e-4,
        input_dim=8,
        n_actions=4,
        mem_size=int(1e6),
        eps_min=1e-2,
        batch_size=64,
        eps_dec=1e-3,
        replace=100,
    )

    if load_checkpoint:
        agent.load_models()

    return agent


def unwrap_observation(observation):
    return observation[0] if isinstance(observation, tuple) else observation


def reset_env(env: gym.Env):
    observation = env.reset()
    return unwrap_observation(observation)


def get_next_state(env: gym.Env, action) -> tuple[None, SupportsFloat, int, dict]:
    # Old GYM API
    # observation_, reward, done, info = env.step(action)

    # New GYM API
    observation_, reward, terminated, truncated, info = env.step(action)
    observation = unwrap_observation(observation_)
    done = int(terminated and truncated)
    return (observation, reward, done, info)


def run_game(env: gym.Env, agent: Agent) -> float:
    done = False
    score = 0.0

    # Iniciamos el entorno en el estado inicial.
    state = reset_env(env)
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = get_next_state(env, action)

        # Guardaríamos una observación y aprenderíamos.
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        # Avanzaríamos al siguiente momento
        state = next_state
        score += float(reward)

    return score


def train_agent(num_games: int, env: gym.Env, agent: Agent, game_sample: int = 100):
    scores = []
    eps_history = []

    for i in range(num_games):
        # ! Ejecutar una partida sobre el entorno
        score = run_game(env, agent)

        # Guardamos los resultados parciales
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 10 == 0:
            avg_score = np.mean(scores[-game_sample:])
            avg_eps = np.mean(eps_history[-game_sample:])
            print(
                f"Episode T={i:4d} | "
                f"Score : {score:1.2f} | "
                f"Avg. score last {game_sample:3d} games : {avg_score:1.3f} | "
                f"Avg. agent eps last {game_sample:3d} games : {avg_eps:1.3f} "
            )

            agent.save_models()


def remove_videos_from(video_folder: str):
    video_dir = glob.glob(os.path.join(video_folder, "*.mp4"))

    for video in video_dir:
        try:
            os.remove(video)
        except:
            print("Error while deleting file : ", video)


def create_videodir(video_folder: str):
    try:
        os.makedirs(video_folder, exist_ok=False)
    except OSError:
        print("Directory already exists!")
        remove_videos_from(video_folder)


def create_video_env(env_id: str, video_folder: str) -> tuple[gym.Env, VideoRecorder]:
    create_videodir(video_folder)

    env = gym.make(id=env_id, render_mode="rgb_array")
    video_recorder = VideoRecorder(
        env, path=f"{video_folder}/render.mp4", enabled=f"{video_folder}/render.mp4" is not None
    )

    return env, video_recorder


def render_game(video_recorder: VideoRecorder, env: gym.Env, agent: Agent) -> float:
    done = False
    score = 0.0

    # Iniciamos el entorno en el estado inicial.
    state = reset_env(env)
    while not done:
        env.unwrapped.render()
        video_recorder.capture_frame()

        action = agent.choose_action(state)
        next_state, reward, done, _ = get_next_state(env, action)

        # Guardaríamos una observación y aprenderíamos.
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        # Avanzaríamos al siguiente momento
        state = next_state
        score += float(reward)

    return score


def generate_playback_video(video_folder: str, env_id: str, agent: Agent) -> float:
    env, video_recorder = create_video_env(env_id, video_folder)
    score = render_game(video_recorder, env, agent)
    env.close()
    video_recorder.close()
    return score


def main(load_checkpoint=False):
    env_id = "LunarLander-v2"
    agent = load_agent(load_checkpoint)

    if not load_checkpoint:
        # Train the agent on the environment, for 500 games.
        env = load_env(env_id)
        train_agent(500, env, agent)
        env.close()

    # Record a video
    test_score = generate_playback_video("videos", env_id, agent)
    print(f"**TEST SCORE: {test_score:1.3f}")


if __name__ == "__main__":
    n = len(sys.argv)
    load_checkpoint = False

    if n == 2 and (sys.argv[1] == "-chkpoint"):
        load_checkpoint = True

    main(load_checkpoint)
