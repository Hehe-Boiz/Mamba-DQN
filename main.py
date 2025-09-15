import gymnasium as gym
import torch
from src.dqn_agent import DQNAgent

# Cấu hình các tham số
CONFIG = {
    "ENV_ID": "ALE/Pong-v5",
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "EPS_START": 0.9,
    "EPS_END": 0.01,
    "EPS_DECAY": 200000,
    "TAU": 0.002,
    "LR": 2.5e-4,
    "TARGET_UPDATE": 10,
    "LEARN_EVERY": 4,
    "MEMORY_SIZE": 40000,
    "NUM_EPISODES": 1000000,
    "MODEL_PATH": "ddqn_per_pong.pth"
}

def make_env(env_id):
    """Hàm tạo môi trường Gymnasium và áp dụng các wrapper cần thiết."""
    env = gym.make(env_id)
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

if __name__ == '__main__':
    env = make_env(CONFIG["ENV_ID"])

    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        batch_size=CONFIG["BATCH_SIZE"],
        gamma=CONFIG["GAMMA"],
        eps_start=CONFIG["EPS_START"],
        eps_end=CONFIG["EPS_END"],
        eps_decay=CONFIG["EPS_DECAY"],
        tau=CONFIG["TAU"],
        lr=CONFIG["LR"],
        target_update=CONFIG["TARGET_UPDATE"],
        learn_every=CONFIG["LEARN_EVERY"],
        memory_size=CONFIG["MEMORY_SIZE"]
    )

    agent.train(env, CONFIG["NUM_EPISODES"], 5)

    agent.save_model(CONFIG["MODEL_PATH"])
    print(f"Model đã được lưu tại {CONFIG['MODEL_PATH']}")

    env.close()
