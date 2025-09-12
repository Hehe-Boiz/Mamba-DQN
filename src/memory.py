import random
import numpy as np
from collections import namedtuple, deque

# Định nghĩa một transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# --- Standard Replay Memory ---
class ReplayMemory(object):
    """Bộ nhớ đệm tiêu chuẩn để lưu trữ và lấy mẫu các transition."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Lưu một transition vào bộ nhớ."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Lấy một batch mẫu ngẫu nhiên từ bộ nhớ."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Trả về kích thước hiện tại của bộ nhớ."""
        return len(self.memory)

# --- Prioritized Experience Replay ---

class SumTree:
    """
    Cấu trúc dữ liệu cây nhị phân mà giá trị của nút cha là tổng của các nút con.
    Được sử dụng để lấy mẫu hiệu quả.
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Cập nhật tổng lên nút gốc."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Tìm mẫu trên nút lá."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Trả về tổng priority."""
        return self.tree[0]

    def add(self, p, data):
        """Lưu trữ priority và mẫu."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Cập nhật priority của một mẫu."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Lấy priority và mẫu."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayMemory:
    """
    Bộ nhớ đệm sử dụng Prioritized Experience Replay.
    Lưu trữ các transition trong một SumTree.
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        """Tính toán priority từ TD error."""
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        """Thêm một mẫu mới vào bộ nhớ với priority tương ứng."""
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """Lấy một batch mẫu, cùng với index và importance sampling weights."""
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        """Cập nhật priority của các mẫu sau khi học."""
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries