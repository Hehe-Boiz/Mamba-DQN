import random
import time
from typing import Callable
from model import DQN
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os


def evaluate(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_episodes: int,
        run_name: str,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.01,
        capture_video: bool = True,
        n_actions: int = None
):
    envs = gym.vector.SyncVectorEnv([
        lambda: make_env(env_id, 0, capture_video, run_name)()
    ])

    if n_actions is None:
        n_actions = envs.single_action_space.n

    obs_shape = envs.single_observation_space.shape
    model = Model(obs_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []

    print(f"Bắt đầu đánh giá với {eval_episodes} màn chơi...")
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = torch.FloatTensor(obs).to(device)
            q_values = model(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"Màn chơi {len(episodic_returns) + 1} kết thúc, điểm số: {info['episode']['r'][0]:.2f}")
                    episodic_returns.append(info['episode']['r'])

        obs = next_obs
        if not capture_video:
            envs.envs[0].render()
            time.sleep(0.02)

    envs.close()
    mean_return = np.mean(episodic_returns)
    print(f"Đánh giá hoàn tất. Điểm trung bình: {mean_return:.2f}")
    return episodic_returns

def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    """
    Hàm tạo môi trường, được tùy chỉnh cho Solaris và để ghi video.
    """
    def thunk():
        render_mode = "rgb_array" if capture_video else "human"
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}", episode_trigger=lambda x: True)

        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk

if __name__ == "__main__":
    evaluate(
        model_path="/home/heheboiz/data/Mamba-DQN/model/ddqn_per_pong_405.pth",
        make_env=make_env,
        env_id="ALE/Pong-v5",
        eval_episodes=5,
        run_name="Pong_evaluation",
        Model=DQN,
        device="cpu",
        capture_video=True
    )
