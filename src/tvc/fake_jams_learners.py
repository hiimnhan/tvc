"""Does TVC hold up under other learners? Credit rules x claim learners on the fake-jams testbed.

python -m tvc.fake_jams_learners [episodes] [seeds]   selfish robots (lam = 0), POGEMA lifelong task
Writes logs/fake_jams_learners.md.
"""

import sys
from dataclasses import replace

from tvc.fake_jams import LOGS, RULES, Config, _train_all_metrics, calibrate
from tvc.util import ProgressPool, ci95

LEARNERS = ("reinforce", "dqn", "ppo")
KEYS = ("lie_rate", "useful_report_rate", "pogema_avg_throughput", "spend")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    base = calibrate(
        replace(Config(), prosocial=0.0)
    )  # the temptation is measured with scripted claims, learner-free
    cfgs = [
        replace(base, rule=r, learner=l, episodes=n, seed=s)
        for l in LEARNERS
        for r in RULES
        for s in range(seeds)
    ]
    with ProgressPool() as pool:
        res = iter(pool.map(_train_all_metrics, cfgs))
    lines = [
        f"### Claim learners (lam = 0, {seeds} seeds, {n} episodes, last-50 mean ± 95% CI)\n",
        "| learner | rule | " + " | ".join(KEYS) + " |",
        "| --- " * (len(KEYS) + 2) + "|",
    ]
    for l in LEARNERS:
        for r in RULES:
            runs = [next(res) for _ in range(seeds)]
            cells = [
                "{:.3f} ± {:.3f}".format(*ci95([x[k] for x in runs])) for k in KEYS
            ]
            lines.append(f"| {l} | {r} | " + " | ".join(cells) + " |")
            print(lines[-1], flush=True)
    LOGS.mkdir(exist_ok=True)
    (LOGS / "fake_jams_learners.md").write_text("\n".join(lines) + "\n")
    print("-> logs/fake_jams_learners.md")
