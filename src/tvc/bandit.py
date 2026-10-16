"""Lean multi-agent bandit testbed for TVC.

N agents search K lognormal arms with private payoffs. Each round a free agent explores (random arm, pays a
cost) or exploits its best known arm. An explorer may post a claim about its find: silent or factor x value.
Free non-explorers (receivers) follow the best claim that beats their own best and lock in for L rounds.
Senders are paid by one of four credit rules; explore and report policies are learned with tabular REINFORCE.

Run: python bandit.py [episodes]   -> prints final metrics, writes figures/bandit_<receiver>.png
"""
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FAC = np.array([0.0, 1.0, 1.1, 1.5, 2.0])  # report actions: silent, truthful, x1.1, x1.5, x2
RULES = ("none", "popularity", "accuracy", "tvc", "shared", "explore", "fine", "pop_fine")
# shared: every agent is paid the group mean payoff; explore: flat credit per exploration;
# fine: lying cost alone (audited false claims are fined, nothing else is paid); pop_fine: popularity credit plus that fine
METRICS = ("inflation", "report", "misled", "welfare", "explore", "credit")
OUT = Path(__file__).resolve().parents[2] / "figures"


@dataclass
class Config:
    runs: int = 32
    agents: int = 10
    arms: int = 50
    rounds: int = 100
    episodes: int = 1000
    rule: str = "tvc"  # none | popularity | accuracy | tvc
    receiver: str = "calibrated"  # naive | calibrated (per-sender decoding) | pooled (one decoding for all senders)
    hetero: bool = False  # True: half the agents can exaggerate at most x1.1 (senders differ in inflation)
    explore_cost: float = 0.5  # x mean value
    lock_in: int = 3
    beta: float = 2.0  # popularity: credit per follow, x mean value
    gamma: float = 1.0  # accuracy: credit per unit of follower's (signed) gain
    tau: float = 0.05  # TVC parameters, as in tvc.TVCParams
    budget: float = 8.0
    eta: float = 2.0
    p_audit: float = 0.2
    fine: float = 8.0
    width: float = 0.05
    explore_pay: float = 3.0  # rule "explore": flat credit per exploration, x mean value (about TVC's spend per exploration)
    scale: float = 1.0  # budget knob: multiplies beta, gamma, and TVC's budget cap (and score weight if scale_eta)
    scale_eta: bool = True
    lr: float = 0.1
    discount: float = 0.9
    trust_lr: float = 0.3
    seed: int = 0


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def x_state(best, vbar):  # own best relative to mean value: 3 buckets
    return np.digitize(best / vbar, [1.0, 2.0])


def r_state(v, vbar):  # value of the find: 4 buckets
    return np.digitize(v / vbar, [0.5, 1.0, 2.0])


def make_policy(cfg):
    B, N = cfg.runs, cfg.agents
    return SimpleNamespace(x=np.zeros((B, N, 3, 2)), r=np.zeros((B, N, 4, len(FAC))),
                           bx=np.zeros((B, N, 3)), br=np.zeros((B, N, 4)))


def credit(cfg, rng, F, sent, fac, theta, b_before, vbar):
    """Credit per sender (B, N). F[b, j, i] = 1 if receiver j follows sender i; theta = true value of i's find."""
    nf = F.sum(1)
    if cfg.rule in ("none", "shared", "explore"):  # these pay through the reward in run_episode, not through credit
        return np.zeros_like(theta)
    if cfg.rule in ("fine", "pop_fine"):  # lying cost alone, or popularity credit plus the same lying cost
        audited = sent & (rng.random(sent.shape) < cfg.p_audit)
        fined = (audited & (np.abs(fac - 1) > cfg.tau)) * cfg.fine * vbar
        return (cfg.scale * cfg.beta * vbar * nf if cfg.rule == "pop_fine" else 0.0) - fined
    if cfg.rule == "popularity":
        return cfg.scale * cfg.beta * vbar * nf
    diff = cfg.lock_in * (theta[:, None, :] - b_before[:, :, None])  # follower j's gain over the lock-in, signed
    if cfg.rule == "accuracy":
        return cfg.scale * cfg.gamma * (F * diff).sum(1)
    verified = np.abs(fac - 1) <= cfg.tau
    cap = cfg.scale * cfg.budget * vbar[:, :, None]
    ext = (F * np.minimum(np.maximum(diff, 0), cap)).sum(1) * verified
    audited = sent & (rng.random(sent.shape) < cfg.p_audit)
    score = np.exp(-np.log(np.where(sent, fac, 1.0)) ** 2 / (2 * cfg.width**2))
    bonus = (audited | (nf > 0)) * (cfg.scale if cfg.scale_eta else 1.0) * cfg.eta * vbar * score
    return ext + bonus - (audited & ~verified) * cfg.fine * vbar  # fine not scaled: honesty margin holds at any scale


def run_episode(cfg, rng, pol, ratio, learn=True):
    """One episode in all runs. Updates pol and ratio in place; returns per-run metrics, each shape (B,)."""
    B, N, K, L = cfg.runs, cfg.agents, cfg.arms, cfg.lock_in
    bi, ai, eye = np.arange(B)[:, None], np.arange(N)[None, :], np.eye(N, dtype=bool)
    values = rng.lognormal(0.0, 1.0, (B, K))
    vbar = values.mean(1, keepdims=True)
    best = values[bi, rng.integers(K, size=(B, N))]  # each agent starts knowing one arm
    lock, cval = np.zeros((B, N), int), np.zeros((B, N))
    allowed = np.zeros((N, len(FAC)))  # report-logit mask: capped agents cannot claim x1.5 or x2
    if cfg.hetero:
        allowed[N // 2:, 3:] = -1e9
    tr ={k: [] for k in ("xs", "xa", "px", "free", "rew", "rs", "ra", "pr", "expl", "cred")}
    s = {k: np.zeros(B) for k in ("explore", "sent", "inflated", "follows", "misled", "welfare", "credit")}

    for _ in range(cfg.rounds):
        free = lock == 0
        xs = x_state(best, vbar)
        px = softmax(pol.x[bi, ai, xs])
        xa = (rng.random((B, N)) < px[..., 1]).astype(int)
        explore = free & (xa == 1)
        v_new = values[bi, rng.integers(K, size=(B, N))]
        u = np.where(~free, cval, np.where(explore, v_new - cfg.explore_cost * vbar, best))
        b_before = best.copy()
        best = np.where(explore, np.maximum(best, v_new), best)

        rs = r_state(v_new, vbar)
        pr = softmax(pol.r[bi, ai, rs] + allowed)
        ra = np.where(explore, (rng.random((B, N, 1)) < pr.cumsum(-1)).argmax(-1), 0)
        sent, fac = ra > 0, FAC[ra]

        dec = (fac * v_new)[:, None, :] / (1.0 if cfg.receiver == "naive" else ratio)
        D = np.where(sent[:, None, :] & ~eye, dec, -np.inf)
        i_star = D.argmax(2)
        follow = free & ~explore & (D.max(2) > b_before)
        F = (np.arange(N)[None, None, :] == i_star[:, :, None]) & follow[:, :, None]
        theta_f = v_new[bi, i_star]

        lock = np.where(follow, L, np.maximum(lock - 1, 0))
        cval = np.where(follow, theta_f, cval)
        best = np.where(follow, np.maximum(best, theta_f), best)
        if cfg.receiver == "calibrated":  # follower sees the truth, updates its decoding of that sender
            ratio[:] = np.where(F, ratio + cfg.trust_lr * (fac[:, None, :] - ratio), ratio)
        elif cfg.receiver == "pooled":  # same update, but one decoding shared across all senders
            r = ratio[:, :, 0]
            ratio[:] = np.where(follow, r + cfg.trust_lr * (fac[bi, i_star] - r), r)[:, :, None]

        c = credit(cfg, rng, F, sent, fac, v_new, b_before, vbar)
        if cfg.rule == "explore":  # flat credit per exploration, whatever is reported
            c = cfg.explore_pay * vbar * explore
        own = u.mean(1, keepdims=True) + 0 * u if cfg.rule == "shared" else u  # shared return: group mean payoff
        for k, v in zip(tr, (xs, xa, px, free, (own + c) / vbar, rs, ra, pr, explore, c / vbar)):
            tr[k].append(v)
        s["explore"] += explore.sum(1)
        s["sent"] += sent.sum(1)
        s["inflated"] += (sent & (fac > 1)).sum(1)
        s["follows"] += follow.sum(1)
        s["misled"] += (follow & (theta_f < b_before)).sum(1)
        s["welfare"] += (u / vbar).sum(1)
        s["credit"] += (np.maximum(c, 0) / vbar).sum(1)

    if learn:
        G = np.zeros((B, N))
        for t in reversed(range(cfg.rounds)):  # explore policy: discounted own return
            G = tr["rew"][t] + cfg.discount * G
            update(pol.x, pol.bx, tr["xs"][t], tr["xa"][t], tr["px"][t], G, tr["free"][t], cfg.lr, bi, ai)
            ret_r = G if cfg.rule == "shared" else tr["cred"][t]  # shared: the report is paid the discounted group return
            update(pol.r, pol.br, tr["rs"][t], tr["ra"][t], tr["pr"][t], ret_r, tr["expl"][t], cfg.lr, bi, ai)

    nT = N * cfg.rounds
    return dict(inflation=s["inflated"] / np.maximum(s["sent"], 1), report=s["sent"] / np.maximum(s["explore"], 1),
                misled=s["misled"] / np.maximum(s["follows"], 1), welfare=s["welfare"] / nT,
                explore=s["explore"] / nT, credit=s["credit"] / np.maximum(s["explore"], 1),
                spend=s["credit"] / nT)  # credit paid per agent-round, x mean value


def update(logits, base, state, action, probs, ret, mask, lr, bi, ai):
    """Tabular REINFORCE step with a per-state running baseline."""
    adv = np.clip(ret - base[bi, ai, state], -10, 10) * mask
    grad = np.eye(probs.shape[-1])[action] - probs
    logits[bi, ai, state] += lr * adv[..., None] * grad
    base[bi, ai, state] += 0.05 * mask * (ret - base[bi, ai, state])


def train(cfg):
    """Returns {metric: array (episodes, runs)}."""
    rng = np.random.default_rng(cfg.seed)
    pol, ratio = make_policy(cfg), np.ones((cfg.runs, cfg.agents, cfg.agents))
    hist = {m: np.zeros((cfg.episodes, cfg.runs)) for m in (*METRICS, "spend")}
    for e in range(cfg.episodes):
        for m, v in run_episode(cfg, rng, pol, ratio).items():
            hist[m][e] = v
    return hist


FRONTIER = {"popularity": dict(rule="popularity"), "accuracy": dict(rule="accuracy"), "tvc": dict(rule="tvc")}
FRONTIER_SMALL = {"popularity": dict(rule="popularity"), "tvc": dict(rule="tvc"),
                  "tvc, small fixed bonus": dict(rule="tvc", eta=0.25, scale_eta=False)}


def frontier(episodes, variants=FRONTIER, name="bandit_frontier.png", scales=(0.25, 0.5, 1.0, 2.0, 4.0), tail=100):
    """Welfare vs credit spent, sweeping each variant's budget knob. Writes figures/<name>."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["tab:red", "tab:orange", "tab:blue", "tab:green", "tab:purple"]
    colors = dict(zip(variants, palette if "accuracy" in variants else ["tab:red", "tab:blue", "tab:green"]))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, receiver in enumerate(("naive", "calibrated")):
        base = train(replace(Config(), rule="none", receiver=receiver, episodes=episodes))
        w0 = base["welfare"][-tail:].mean()
        print(f"\n{receiver} receivers; no-credit welfare {w0:.3f}")
        for rule, c in colors.items():
            pts = []
            for s in scales:
                h = train(replace(Config(), receiver=receiver, episodes=episodes, scale=s, **variants[rule]))
                v = {m: h[m][-tail:].mean(0) for m in ("spend", "welfare", "explore", "inflation")}
                pts.append({m: (x.mean(), 1.96 * x.std() / np.sqrt(len(x))) for m, x in v.items()})
                print(f"  {rule:<22} scale {s:<5} " + "  ".join(f"{m} {a:.3f}±{e:.3f}" for m, (a, e) in pts[-1].items()))
            for ax, m in zip(axes[row], ("welfare", "explore", "inflation")):
                ax.errorbar([p["spend"][0] for p in pts], [p[m][0] for p in pts], yerr=[p[m][1] for p in pts],
                            color=c, marker="o", capsize=3, label=rule)
        axes[row, 0].axhline(w0, color="grey", ls=":", label="no credit")
        for ax, m in zip(axes[row], ("Welfare", "Exploration rate", "Inflation")):
            ax.set_title(f"{m} vs credit spent ({receiver} receivers)", fontsize=10)
            ax.set_xlabel("Credit paid per agent-round (× mean value)")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle(f"Budget frontier: each rule's budget knob swept over scales {scales} "
                 f"(last {tail} episodes, 95% CI over runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


CASES = {
    "Heterogeneous senders\n(pooled receivers)": dict(hetero=True, receiver="pooled"),
    "Heterogeneous senders\n(per-sender receivers)": dict(hetero=True, receiver="calibrated"),
    "Costly following, lock-in 10\n(naive receivers)": dict(lock_in=10, receiver="naive"),
    "Costly following, lock-in 10\n(calibrated receivers)": dict(lock_in=10, receiver="calibrated"),
}


def cases(episodes, tail=100):
    """Each credit rule in settings where hype might hurt. Writes figures/bandit_cases.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ("welfare", "misled", "inflation", "spend")
    colors = dict(none="grey", popularity="tab:red", accuracy="tab:orange", tvc="tab:blue", shared="tab:green", explore="tab:brown", fine="tab:pink", pop_fine="tab:olive")
    fig, axes = plt.subplots(len(show), len(CASES), figsize=(16, 11), sharex=True)
    for col, (case, over) in enumerate(CASES.items()):
        print(f"\n{case.replace(chr(10), ' ')}")
        for k, rule in enumerate(RULES):
            h = train(replace(Config(), rule=rule, episodes=episodes, **over))
            print(f"  {rule:<10} " + "  ".join(
                f"{m} {h[m][-tail:].mean():.3f}±{1.96 * h[m][-tail:].mean(0).std() / np.sqrt(h[m].shape[1]):.3f}"
                for m in show))
            for row, m in enumerate(show):
                v = h[m][-tail:].mean(0)
                axes[row, col].bar(k, v.mean(), yerr=1.96 * v.std() / np.sqrt(len(v)), color=colors[rule], capsize=3)
        axes[0, col].set_title(case, fontsize=10)
    for row, m in enumerate(show):
        axes[row, 0].set_ylabel(dict(welfare="Welfare", misled="Misled follows", inflation="Inflation",
                                     spend="Credit per agent-round")[m])
        for ax in axes[row]:
            ax.grid(axis="y", alpha=0.3)
    for ax in axes[-1]:
        ax.set_xticks(range(len(RULES)))
        ax.set_xticklabels(RULES, fontsize=8)
    for ax in axes[0]:  # zoom welfare to the range that shows differences
        lo = min(b.get_height() for b in ax.patches)
        ax.set_ylim(lo * 0.9, None)
    fig.suptitle(f"Where might hype hurt? Credit rules at default budgets (last {tail} episodes, 95% CI over runs)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "bandit_cases.png", dpi=150)
    plt.close(fig)


def plot(results, receiver, path, window=50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = dict(inflation="Inflation (share of claims exaggerated)", report="Report rate (claims / explorations)",
                  misled="Misled follows (tip worse than own best)", welfare="Welfare (own payoff / mean value)",
                  explore="Exploration rate", credit="Credit paid per exploration")
    colors = dict(none="grey", popularity="tab:red", accuracy="tab:orange", tvc="tab:blue", shared="tab:green", explore="tab:brown", fine="tab:pink", pop_fine="tab:olive")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    kernel = np.ones(window) / window
    for ax, m in zip(axes.flat, METRICS):
        for rule, hist in results.items():
            h = hist[m]
            mean = np.convolve(h.mean(1), kernel, "valid")
            se = np.convolve(h.std(1), kernel, "valid") / np.sqrt(h.shape[1])
            x = np.arange(len(mean)) + window
            ax.plot(x, mean, color=colors[rule], label=rule)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=colors[rule], alpha=0.2)
        ax.set_title(titles[m], fontsize=10)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Multi-agent bandit, {receiver} receivers: learning curves by credit rule "
                 f"({window}-episode moving average, 95% CI over runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    mode, n = sys.argv[1:2], int(sys.argv[2]) if len(sys.argv) > 2 else 600
    if mode == ["frontier"]:  # python bandit.py frontier [episodes]
        sys.exit(frontier(n))
    if mode == ["frontier_small"]:  # same, with TVC's score bonus small and not scaled
        sys.exit(frontier(n, FRONTIER_SMALL, "bandit_frontier_small_bonus.png"))
    if mode == ["cases"]:  # heterogeneous senders, costly following
        sys.exit(cases(n))
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else Config.episodes
    for receiver in ("naive", "calibrated"):
        results = {rule: train(replace(Config(), rule=rule, receiver=receiver, episodes=episodes)) for rule in RULES}
        plot(results, receiver, OUT / f"bandit_{receiver}.png")
        print(f"\n{receiver} receivers, last 100 episodes (mean ± 95% CI over runs)")
        for rule, hist in results.items():
            cells = []
            for m in METRICS:
                v = hist[m][-100:].mean(0)
                cells.append(f"{m} {v.mean():.3f}±{1.96 * v.std() / np.sqrt(len(v)):.3f}")
            print(f"  {rule:<10} " + "  ".join(cells))
