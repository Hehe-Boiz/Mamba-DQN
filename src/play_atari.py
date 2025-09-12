import random
import time
from typing import Callable
from dqn_agent import DQN
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque #

def evaluate(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_episodes: int,
        run_name: str,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.01,  # Giảm epsilon để agent chủ yếu dùng chính sách đã học
        capture_video: bool = True
):
    # Dùng lambda để truyền tham số vào make_env
    envs = gym.vector.SyncVectorEnv([
        lambda: make_env(env_id, 0, capture_video, run_name)
    ])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []

    print(f"Bắt đầu đánh giá với {eval_episodes} màn chơi...")
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(obs)  # obs đã là numpy array, model sẽ xử lý
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)

        # Xử lý thông tin khi một màn chơi kết thúc
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"Màn chơi {len(episodic_returns) + 1} kết thúc, điểm số: {info['episode']['r']:.2f}")
                    episodic_returns.append(info['episode']['r'])

        obs = next_obs
        # Nếu không ghi video, ta có thể render trực tiếp để xem
        if not capture_video:
            envs.render()
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
        # THAY ĐỔI: render_mode='rgb_array' để ghi video
        render_mode = "rgb_array" if capture_video else "human"
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Gói môi trường để ghi video nếu cần
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # THAY ĐỔI: Áp dụng các bước xử lý của Atari
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk

if __name__ == "__main__":
    evaluate(
        model_path="dqn_solaris.pth",
        make_env=make_env,
        env_id="ALE/Solaris-v5",
        eval_episodes=5,
        run_name="solaris_evaluation",
        Model=DQN,
        device="cpu",
        capture_video=True
    )
