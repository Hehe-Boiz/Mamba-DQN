import math
import random
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import DQN
# Thay đổi import để sử dụng PrioritizedReplayMemory
from memory import PrioritizedReplayMemory, Transition
from plotting import *


class DQNAgent_per:
    def __init__(self, input_shape, n_actions,
                 batch_size=64, gamma=0.99, eps_start=0.9, eps_end=0.01,
                 eps_decay=10000, tau=0.005, lr=3e-4, target_update=10,
                 learn_every=4, memory_size=30000):

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
        # Sử dụng PrioritizedReplayMemory
        self.memory = PrioritizedReplayMemory(memory_size)
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
        """Tính toán TD error ban đầu và lưu transition vào bộ nhớ."""
        # Chuyển đổi sang tensor để tính toán
        if not torch.is_tensor(reward):
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            current_q = self.policy_net(state).gather(1, action)

            # Tính Q-value mục tiêu (DQN thuần túy)
            next_q = 0.0
            if next_state is not None:
                next_q = self.target_net(next_state).max(1).values.item()
            target_q = reward + (self.gamma * next_q)
            # TD error
            error = abs(current_q - target_q).item()
        # Lưu vào bộ nhớ
        self.memory.add(error, (state, action, next_state, reward))

    def optimize_model(self):
        """Tối ưu hóa policy network bằng PER."""
        if len(self.memory) < self.batch_size:
            return

        # Lấy mẫu từ bộ nhớ
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch = Transition(zip(*mini_batch))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Xử lý các trạng thái không phải cuối cùng
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Tính Q(s_t, a) hiện tại
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Tính V(s_{t+1}) cho tất cả các next states (DQN thuần túy)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Sử dụng target network để lấy max Q-value (DQN thuần túy)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Tính Q-value mục tiêu
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Tính TD errors mới để cập nhật priorities
        errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i][0])

        # Tính loss có trọng số (importance sampling)
        is_weights_t = torch.FloatTensor(is_weights).to(self.device)
        criterion = torch.nn.SmoothL1Loss(reduction='none')  # Để nhân với is_weights
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        weighted_loss = (is_weights_t * loss.squeeze()).mean()

        # Tối ưu hóa
        self.optimizer.zero_grad()
        weighted_loss.backward()
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

    def evaluate_agent(self, env, n_episodes=30, eval_epsilon=0.05):
        """
        Chạy agent trong n_episodes với epsilon cố định để đánh giá hiệu suất.
        Không thực hiện training trong quá trình này.
        """
        self.policy_net.eval()  # Chuyển model sang chế độ đánh giá
        total_rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            state = torch.from_numpy(np.array(state, copy=False)).float().to(self.device).unsqueeze(0)
            episode_reward = 0
            done = False
            while not done:
                if random.random() > eval_epsilon:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1).indices.view(1, 1)
                else:
                    action = torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

                observation, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                done = terminated or truncated
                if not done:
                    state = torch.from_numpy(np.array(observation, copy=False)).float().to(self.device).unsqueeze(0)
            total_rewards.append(episode_reward)

        self.policy_net.train()  # Chuyển model về lại chế độ huấn luyện
        return sum(total_rewards) / n_episodes

    def calculate_avg_q(self, states):
        """
        Tính giá trị Q tối đa trung bình trên một tập các trạng thái cố định.
        """
        self.policy_net.eval()  # Chuyển model sang chế độ đánh giá
        with torch.no_grad():
            # Lấy giá trị Q tối đa cho mỗi trạng thái
            max_q_values = self.policy_net(states).max(1).values
            # Tính trung bình và trả về dưới dạng số
            average_q = max_q_values.mean().item()
        self.policy_net.train()  # Chuyển model về lại chế độ huấn luyện
        return average_q

    def train(self, env, num_episodes, eval_every_episodes=20):
        """Vòng lặp huấn luyện chính."""

        fixed_states_list = []
        state, _ = env.reset()
        while len(fixed_states_list) < 1000:
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            fixed_states_list.append(torch.from_numpy(np.array(state, copy=False)).float().to(self.device).unsqueeze(0))
            state = obs
            if term or trunc:
                state, _ = env.reset()
        fixed_states_tensor = torch.cat(fixed_states_list, 0)
        print(f"Collected {fixed_states_tensor.shape[0]} states.")

        avg_rewards_list = []
        avg_q_values_list = []
        evaluation_points = []

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

                # Sử dụng hàm store_transition
                self.store_transition(state, action, next_state, reward_t)

                state = next_state

                if self.steps_done % self.learn_every == 0:
                    self.optimize_model()

                if self.steps_done % self.target_update == 0:
                    self.update_target_net()

                if done:
                    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                        -1. * self.steps_done / self.eps_decay)
                    print(
                        f"Episode {i_episode:3d} | Steps: {t + 1:4d} | Reward: {total_reward:8.2f} | Epsilon: {eps_threshold:.3f}")
                    break
            if i_episode % eval_every_episodes == 0:
                avg_reward = self.evaluate_agent(env)
                avg_q = self.calculate_avg_q(fixed_states_tensor)

                avg_rewards_list.append(avg_reward)
                avg_q_values_list.append(avg_q)
                evaluation_points.append(i_episode)

                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                    -1. * self.steps_done / self.eps_decay)
                print(
                    f"Episode {i_episode:4d} | Avg Reward (10 ep): {avg_reward:8.2f} | Avg Max Q: {avg_q:8.2f} | Epsilon: {eps_threshold:.3f}")

                plot_training_progress(evaluation_points, avg_rewards_list, avg_q_values_list)

        print('Huấn luyện hoàn tất')
        plot_training_progress(evaluation_points, avg_rewards_list, avg_q_values_list, show_result=True)

    def save_model(self, path):
        """Lưu trọng số của policy network."""
        torch.save(self.policy_net.state_dict(), path)