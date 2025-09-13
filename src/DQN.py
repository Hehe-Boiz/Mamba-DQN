import math
import random
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import DQN
from src.memory import ReplayMemory, Transition
from src.plotting import plot_durations


class DQNAgent_er:
    def __init__(self, input_shape, n_actions,
                 batch_size=32, gamma=0.99, eps_start=0.9, eps_end=0.01,
                 eps_decay=10000, tau=0.005, lr=3e-4, target_update=10,
                 learn_every=4, memory_size=10000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.target_update = target_update
        self.learn_every = learn_every

        self.policy_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0

    def select_action(self, state, env):
        """Chọn hành động theo chiến lược epsilon-greedy."""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward):
        """THAY ĐỔI: Lưu transition vào bộ nhớ một cách đơn giản."""
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        """THAY ĐỔI: Tối ưu hóa policy network bằng ER thông thường."""
        if len(self.memory) < self.batch_size:
            return

        # Lấy mẫu từ bộ nhớ
        transitions = self.memory.sample(self.batch_size)
        # Chuyển đổi batch-array của Transitions thành Transition của batch-arrays.
        batch = Transition(*zip(*transitions))

        # Xử lý các trạng thái không phải cuối cùng
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Tính Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Tính V(s_{t+1}) cho tất cả các next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Tính Q-value mục tiêu
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # THAY ĐỔI: Tính loss thông thường
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Tối ưu hóa
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """Cập nhật trọng số của target network."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                         target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, env, num_episodes):
        """Vòng lặp huấn luyện chính."""
        episode_durations = []
        episode_rewards = []

        for i_episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.from_numpy(np.array(state, copy=False)).float().to(self.device).unsqueeze(0)
            total_reward = 0

            for t in count():
                action = self.select_action(state, env)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                total_reward += reward
                reward_t = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = None if terminated else torch.from_numpy(np.array(observation, copy=False)).float().to(
                    self.device).unsqueeze(0)

                # Sử dụng hàm store_transition đã được cập nhật
                self.store_transition(state, action, next_state, reward_t)

                state = next_state

                if self.steps_done % self.learn_every == 0:
                    self.optimize_model()

                if self.steps_done % self.target_update == 0:
                    self.update_target_net()

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)

                    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                        -1. * self.steps_done / self.eps_decay)
                    print(
                        f"Episode {i_episode:3d} | Steps: {t + 1:4d} | Reward: {total_reward:8.2f} | Epsilon: {eps_threshold:.3f}")

                    plot_durations(episode_rewards, episode_durations)
                    break

        print('Huấn luyện hoàn tất')
        plot_durations(episode_rewards, episode_durations, show_result=True)

    def save_model(self, path):
        """Lưu trọng số của policy network."""
        torch.save(self.policy_net.state_dict(), path)