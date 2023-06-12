from typing import SupportsFloat
import glob
import os
import sys
import argparse

import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import torch as T

from src.agent import Agent


def load_env(env_id: str) -> gym.Env:
    return gym.make(id=env_id)


def load_agent(
    chdir,
    input_dim,
    n_actions,
    load_checkpoint: bool = False,
    device: T.device | str = "cpu",
) -> Agent:
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=5e-4,
        input_dim=input_dim,
        n_actions=n_actions,
        mem_size=int(1e6),
        eps_min=1e-2,
        batch_size=64,
        eps_dec=1e-4,
        replace=100,
        device=device,
        chkpt_dir=chdir,
    )

    if load_checkpoint:
        agent.load_model()

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


def run_training_step(env: gym.Env, agent: Agent) -> float:
    done = False
    score = 0.0

    # Iniciamos el entorno en el estado inicial.
    state = reset_env(env)
    while not done:
        action = agent.choose_action(state, is_training=True)
        next_state, reward, done, _ = get_next_state(env, action)

        # Guardaríamos una observación y aprenderíamos.
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        # Avanzaríamos al siguiente momento
        state = next_state
        score += float(reward)

    return score


def train_agent(
    num_games: int,
    env: gym.Env,
    agent: Agent,
    game_sample: int = 100,
    save_period: int = 10,
):
    score_stack = []
    prev_avg_score = -float("inf")

    for i in range(1, num_games + 1):
        score = run_training_step(env, agent)

        # Guardamos los resultados parciales
        score_stack.append(score)
        avg_score = np.mean(score_stack[-game_sample:])

        print(
            f"Episode T={i:4d} / {num_games} | "
            f"Score : {score:1.2f} | "
            f"Agent eps. : {agent.eps:1.3f} | "
            f"Avg. score, last {game_sample:3d} games : {avg_score:1.3f}"
        )

        if i % save_period == 0:
            if prev_avg_score < avg_score:
                print("Saving model...")
                agent.save_model()

                prev_avg_score = avg_score
            else:
                print(f"Current score: {avg_score:1.3f} < {prev_avg_score:1.3f}")
                print("Restoring prev. model...")
                agent.load_model()

                avg_score = prev_avg_score
                del score_stack[-save_period:]


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


def create_video_env(env_id: str, video_path: str) -> tuple[gym.Env, VideoRecorder]:
    env = gym.make(id=env_id, render_mode="rgb_array")
    video_recorder = VideoRecorder(
        env,
        path=video_path,
        enabled=video_path is not None,
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
    create_videodir(video_folder)
    video_path = f'{video_folder}/{env_id}_render.mp4'
    
    env, video_recorder = create_video_env(env_id, video_path)
    score = render_game(video_recorder, env, agent)
    env.close()
    video_recorder.close()
    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--actions", type=int)
    parser.add_argument("--num-games", type=int, default=150)
    parser.add_argument("--avg-sample", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--checkpoint", action='store_true')
    # parser.add_argument("--continuous", action='store_true')

    args = parser.parse_args()
    env_id = args.env
    num_games = args.num_games
    game_sample = args.avg_sample
    save_period = args.save_period

    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    # Creamos el agente.
    agent = load_agent(
        chdir=f"tmp/{env_id}",
        input_dim=args.input_dim,
        n_actions=args.actions,
        load_checkpoint=args.checkpoint,
        device=device,
    )

    if not args.checkpoint:
        # Train the agent on the environment, for 500 games.
        print(f"Begin train for num_games={num_games}")
        env = load_env(env_id)
        train_agent(num_games, env, agent, game_sample, save_period)
        env.close()
    else:
        print("Loading checkpoint...")

    # Record a video
    test_score = generate_playback_video(f"tmp/{env_id}", env_id, agent)
    print(f"**TEST SCORE: {test_score:1.3f}")


if __name__ == "__main__":
    main()
