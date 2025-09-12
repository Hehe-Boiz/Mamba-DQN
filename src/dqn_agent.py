import math
import random
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.models import DQN
# Thay đổi import để sử dụng PrioritizedReplayMemory
from src.memory import PrioritizedReplayMemory, Transition
from src.plotting import plot_durations


class DQNAgent:
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
        state_t = torch.cat([state])
        action_t = torch.cat([action])
        reward_t = torch.cat([reward])
        
        # Dự đoán Q-value hiện tại
        current_q = self.policy_net(state_t).gather(1, action_t)

        # Tính Q-value mục tiêu
        next_q = 0.0
        if next_state is not None:
            with torch.no_grad():
                next_q = self.target_net(torch.cat([next_state])).max(1).values.detach()
        
        target_q = reward_t + (self.gamma * next_q)
        
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
        mini_batch = list(zip(*mini_batch)) # Unzip

        # Unpack và chuyển đổi sang tensor
        states = [t[0] for t in mini_batch[0]]
        actions = [t[0] for t in mini_batch[1]]
        next_states = [t[0] for t in mini_batch[2]]
        rewards = [t[0] for t in mini_batch[3]]

        state_batch = torch.cat(states)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        # Xử lý các trạng thái không phải cuối cùng
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])

        # Tính Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Tính V(s_{t+1}) cho tất cả các next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
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
        criterion = torch.nn.SmoothL1Loss(reduction='none') # Để nhân với is_weights
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

                next_state = None if terminated else torch.from_numpy(np.array(observation, copy=False)).float().to(self.device).unsqueeze(0)
                
                # Thay đổi: Sử dụng hàm store_transition mới
                self.store_transition(state, action, next_state, reward_t)
                
                state = next_state

                if self.steps_done % self.learn_every == 0:
                    self.optimize_model()

                if self.steps_done % self.target_update == 0:
                    self.update_target_net()

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)
                    
                    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                    print(f"Episode {i_episode:3d} | Steps: {t + 1:4d} | Reward: {total_reward:8.2f} | Epsilon: {eps_threshold:.3f}")
                    
                    plot_durations(episode_rewards, episode_durations)
                    break
        
        print('Huấn luyện hoàn tất')
        plot_durations(episode_rewards, episode_durations, show_result=True)

    def save_model(self, path):
        """Lưu trọng số của policy network."""
        torch.save(self.policy_net.state_dict(), path)