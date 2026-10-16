import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    """Policy network: claim state -> logits over {silent, claim}."""

    def __init__(self, n_states=4, n_actions=2, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        """x is (batch, n_states) one-hot; returns logits of shape (batch, n_actions)."""
        return self.layers(x)

    def dist(self, x):
        """Categorical over the actions, for log_prob and entropy in the loss."""
        return torch.distributions.Categorical(logits=self(x))


class Critic(nn.Module):
    """Value network: claim state -> V(state), the baseline for the advantage."""

    def __init__(self, n_states=4, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """x is (batch, n_states) one-hot; returns values of shape (batch,)."""
        return self.layers(x).squeeze(-1)


class Policy:
    def __init__(self, cfg, lr=3e-4, clip=0.2, epochs=10, entropy=0.01):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.n, self.clip, self.epochs, self.entropy = cfg.agents, clip, epochs, entropy
        self.actors = [
            Actor() for _ in range(self.n)
        ]  # one pair per robot: independent learners
        self.critics = [Critic() for _ in range(self.n)]
        self.opts = [
            torch.optim.Adam([*a.parameters(), *c.parameters()], lr=lr)
            for a, c in zip(self.actors, self.critics)
        ]
        self.eye = torch.eye(4)
        self.logits = np.zeros((self.n, 4, 2))  # uniform start, as in REINFORCE
        self._refresh()

    def _refresh(self):
        """Recompute self.logits from the Actors, for all 4 states of every robot."""
        with torch.no_grad():
            self.logits = torch.stack([a(self.eye) for a in self.actors]).numpy()

    def update(self, batch):
        """batch: list of (robot, state, action, probs, reward) from one episode."""
        per = {}
        for i, state, action, probs, r in batch:
            per.setdefault(i, []).append(
                (state, action, np.log(probs[action]), np.clip(r, -10, 10))
            )
        for i, rows in per.items():
            cols = list(zip(*rows))
            s = torch.as_tensor(cols[0], dtype=torch.long)
            a = torch.as_tensor(cols[1], dtype=torch.long)
            old = torch.as_tensor(cols[2], dtype=torch.float32)
            r = torch.as_tensor(cols[3], dtype=torch.float32)
            x = self.eye[s]
            with (
                torch.no_grad()
            ):  # one-step decisions: return = reward, advantage = reward - V(state)
                adv = r - self.critics[i](x)
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.epochs):
                dist = self.actors[i].dist(x)
                ratio = (dist.log_prob(a) - old).exp()
                surrogate = torch.min(
                    ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv
                )
                value_loss = ((self.critics[i](x) - r) ** 2).mean()
                loss = (
                    -surrogate.mean()
                    + 0.5 * value_loss
                    - self.entropy * dist.entropy().mean()
                )
                self.opts[i].zero_grad()
                loss.backward()
                self.opts[i].step()
        self._refresh()
