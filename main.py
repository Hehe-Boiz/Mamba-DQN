import gymnasium as gym
import torch
import ale_py
from src.dqn_er import DQNAgent_er

# Cấu hình các tham số
CONFIG = {
    "ENV_ID": "ALE/Pong-v5",
    "BATCH_SIZE": 32,
    "GAMMA": 0.99,
    "EPS_START": 0.9,
    "EPS_END": 0.01,
    "EPS_DECAY": 10000,
    "EVAL_EVERY_EPISODE": 10,
    "TAU": 0.005,
    "LR": 2.5e-4,
    "TARGET_UPDATE": 30,
    "LEARN_EVERY": 4,
    "MEMORY_SIZE": 10000,
    "NUM_EPISODES": 100,
    "MODEL_PATH": "dqn_er.pth"
}

def make_env(env_id):
    """Hàm tạo môi trường Gymnasium và áp dụng các wrapper cần thiết."""
    env = gym.make(env_id)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

if __name__ == '__main__':
    env = make_env(CONFIG["ENV_ID"])

    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent_er(
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

    agent.train(env, CONFIG["NUM_EPISODES"], CONFIG["EVAL_EVERY_EPISODE"])

    agent.save_model(CONFIG["MODEL_PATH"])
    print(f"Model đã được lưu tại {CONFIG['MODEL_PATH']}")

    env.close()
