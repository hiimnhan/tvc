import numpy as np
import torch
from torch import nn
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, n_states=4, n_actions=2, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


def epsilon_probs(q, epsilon):
    """q has shape (robots, states, 2). Returns the probability of each action."""
    best = q == q.max(
        -1, keepdims=True
    )  # mask of the best actions; exact ties share the greedy mass
    return epsilon / q.shape[-1] + (1 - epsilon) * best / best.sum(-1, keepdims=True)


class Policy:
    def __init__(
        self,
        cfg,
        lr=1e-3,
        episodes=100,
        eps_start=1.0,
        eps_end=0.0,
        buffer=10_000,
        batch_size=64,
        steps=10,
    ):
        """episodes: how many episodes epsilon takes to fall from eps_start to eps_end."""
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.n = cfg.agents
        self.rng = np.random.default_rng(cfg.seed)
        self.robots = [QNetwork() for _ in range(self.n)]  # independent learners
        self.optimisers = [
            torch.optim.Adam(net.parameters(), lr=lr) for net in self.robots
        ]
        self.replay_buffers = [
            deque(maxlen=buffer) for _ in range(self.n)
        ]  # (state, action, reward) per robot
        self.eye = torch.eye(4)
        self.episodes = episodes
        self.global_step = 0  # episodes seen so far
        self.eps_start, self.eps_end = eps_start, eps_end
        self.buffer, self.batch_size, self.steps = buffer, batch_size, steps
        self.logits = np.zeros((self.n, 4, 2))  # uniform start, as in REINFORCE
        self._refresh()

    def epsilon(self):
        frac = min(self.global_step / self.episodes, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _refresh(self):

        with torch.no_grad():
            q = torch.stack([net(self.eye) for net in self.robots]).numpy()  # (n, 4, 2)
        self.logits = np.log(np.maximum(epsilon_probs(q, self.epsilon()), 1e-12))  # floor: log(0) at eps = 0

    def update(self, batch):

        for i, state, action, _, r in batch:
            self.replay_buffers[i].append((state, action, float(np.clip(r, -10, 10))))
        for i, net in enumerate(self.robots):
            if not self.replay_buffers[i]:
                continue
            data = np.array(self.replay_buffers[i])
            for _ in range(self.steps):
                rows = self.rng.integers(len(data), size=self.batch_size)
                s = torch.as_tensor(data[rows, 0], dtype=torch.long)
                a = torch.as_tensor(data[rows, 1], dtype=torch.long)
                r = torch.as_tensor(data[rows, 2], dtype=torch.float32)
                q = net(self.eye[s]).gather(1, a[:, None])[:, 0]
                loss = ((q - r) ** 2).mean()
                self.optimisers[i].zero_grad()
                loss.backward()
                self.optimisers[i].step()
        self.global_step += 1
        self._refresh()
