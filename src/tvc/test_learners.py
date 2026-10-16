"""DQN and PPO learn a one-step claim problem: claim pays +1 in state 0 and -1 in state 2, silence pays 0."""

import importlib
from types import SimpleNamespace

import numpy as np


def check(name):
    pol = importlib.import_module(f"tvc.fake_jams_{name}").Policy(SimpleNamespace(agents=2, seed=0))
    rng = np.random.default_rng(0)
    for _ in range(150):
        batch = []
        for i in range(2):
            for _ in range(20):
                state = int(rng.choice([0, 2]))
                z = pol.logits[i, state]
                probs = np.exp(z - z.max())
                probs /= probs.sum()
                action = int(rng.random() < probs[1])
                batch.append((i, state, action, probs, (1.0 if state == 0 else -1.0) * action))
        pol.update(batch)
    p = np.exp(pol.logits) / np.exp(pol.logits).sum(-1, keepdims=True)
    assert (p[:, 0, 1] > 0.8).all() and (p[:, 2, 1] < 0.2).all(), (name, p[:, [0, 2], 1])


if __name__ == "__main__":
    for name in ("dqn", "ppo"):
        check(name)
        print(name, "ok")
