from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tvc.tvc import TVCParams
from tvc.pogema_env import (
    doorway_zones,
    jammed,
    make_env,
    load_map,
    next_action,
    route_doorways,
    summarize,
)
from tvc.util import ProgressPool, bfs_dist, chebyshev_dist, ci95

RULES = ("none", "popularity", "accuracy", "fine", "pop_fine", "tvc")


@dataclass
class Config:
    env: str = "rooms"  # environment = name of a text map in maps/ (rooms | rooms4)
    rooms: int = 3  # rooms x rooms layout
    room_size: int = 5
    agents: int = 12
    obs_radius: int = 3
    steps: int = 256
    episodes: int = 300
    seed: int = 0
    # jam truth
    jam_radius: int = 2  # k: cells around a doorway that count as its queue
    jam_queue: int = (
        3  # q: agents within k walking steps for the doorway to count as jammed
    )
    avoid_steps: int = 20  # how long a believer avoids a claimed doorway
    claim_ttl: int = 20  # claim expires after this many steps
    # receivers
    receiver: str = "naive"  # naive | sceptical (trust per sender, updated on checks)
    trust_lr: float = 0.3
    listen: float = 1.0  # rho: chance a receiver acts on any claim at all (0 = deaf: information changes nothing)
    # credit
    rule: str = "tvc"
    beta: float = 1.0  # popularity: credit per avoider
    gamma: float = 1.0  # accuracy: credit per step saved (negative for fake jams)
    tvc_params: TVCParams = field(default_factory=TVCParams)
    # mixed motive
    prosocial: float = (
        0.3  # lam: weight on others' mean deliveries (0 selfish, 1 cooperative)
    )
    report_cost: float = 0.05  # c: cost per posted claim, so silence is the default
    on_target: str = "restart"  # POGEMA task: 'restart' = lifelong MAPF (throughput), 'finish' = one-shot (CSR, ISR, SoC)
    # private effect of one claim per believer, in deliveries; set by calibrate() from hand-set runs
    own_gain: float = (
        0.0  # sender's own gain when a believer leaves a doorway on the sender's route
    )
    harm: float = (
        0.0  # drop in the others' mean deliveries per believer of a fake claim
    )
    benefit: float = (
        0.0  # rise in the others' mean deliveries per believer of a true claim
    )
    # learning (claim policy only; movement is scripted BFS)
    lr: float = 0.1
    learner: str = (
        "reinforce"  # reinforce | dqn | ppo (fake_jams_dqn.py, fake_jams_ppo.py)
    )
    learner_args: dict = field(default_factory=dict)  # keyword arguments of the learner's Policy (hyperparameters)


# ---------------------------------------------------------------- steps 5-7: claims, receivers, checks


@dataclass
class Claim:
    sender: int
    doorway: int  # index into doorways
    t: int
    is_true: bool  # jammed(doorway) at posting time (simulator knows; agents do not)
    audited: bool
    avoiders: list = field(
        default_factory=list
    )  # [(receiver j, detour steps, wait avoided)]
    considers: set = field(default_factory=set)  # set of agents consider this claim
    checked: bool = False  # revealed by a passer-by or an audit


def make_policy(cfg) -> SimpleNamespace:
    if (
        cfg.learner != "reinforce"
    ):  # learners with an update(batch) method; imported lazily because they need torch
        import importlib

        return importlib.import_module(f"tvc.fake_jams_{cfg.learner}").Policy(cfg, **cfg.learner_args)
    # claim policy per agent, per visible doorway: state = 2 * on_my_route + jammed (4 cells)
    #   0: elsewhere, free   1: elsewhere, jammed (useful report)   2: my route, free (selfish lie)   3: my route, jammed
    # logits over {silent, claim jam}
    return SimpleNamespace(
        logits=np.zeros((cfg.agents, 4, 2)), base=np.zeros((cfg.agents, 4))
    )


def post_claims(
    t,
    visible_doors,
    my_route_doors,
    truth,
    pol,
    cfg,
    rng,
    board,
    forced=None,
    prev_visible=None,
) -> list[tuple[int, int, int, np.ndarray]]:
    """Each agent decides, for each doorway it can see, whether to post "doorway d is jammed"."""
    active = {(c.sender, c.doorway) for c in board if t - c.t <= cfg.claim_ttl}
    decisions = []
    for i, doors in enumerate(visible_doors):
        seen_before = prev_visible[i] if prev_visible is not None else set()
        for d in doors:
            if (i, d) in active or d in seen_before:
                continue
            # state (on my route? / real jammed?):
            # 0 elsewhere+free   1 elsewhere+jammed (useful report)
            # 2 my route+free (selfish lie)   3 my route+jammed
            state = 2 * int(d in my_route_doors[i]) + int(truth[d])
            z = pol.logits[i, state]
            # softmax
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            action = int(forced(i, state)) if forced else int(rng.random() < probs[1])
            decisions.append((i, state, action, probs))
            if action == 1:
                board.append(
                    Claim(
                        sender=i,
                        doorway=d,
                        t=t,
                        is_true=bool(truth[d]),
                        audited=bool(rng.random() < cfg.tvc_params.p_audit),
                    )
                )
        if prev_visible is not None:
            prev_visible[i] = set(doors)
    return decisions


def adopt_claims(
    t,
    board,
    agents_xy,
    avoid,
    trust,
    dists_through,
    dists_around,
    my_route_doors,
    queue_at,
    cfg,
    rng,
) -> None:
    """Receivers whose route passes a claimed doorway decide whether to believe it and route around it."""
    N = len(agents_xy)  # number of agents
    for c in board:
        if t - c.t > cfg.claim_ttl:
            continue
        d = c.doorway
        for i in range(N):
            if i == c.sender:
                continue
            if i in c.considers:
                continue
            if d not in my_route_doors[i]:
                continue
            if avoid[i, d] > 0:
                continue
            believe = (
                True if cfg.receiver == "naive" else rng.random() < trust[i, c.sender]
            )
            c.considers.add(i)
            if cfg.listen < 1 and rng.random() >= cfg.listen:
                believe = False  # deaf this time; drawn only when rho < 1, so rho = 1 reproduces old runs
            if believe:
                detour = dists_around(i, d) - dists_through[i]
                if detour == np.inf:
                    continue
                wait_avoided = queue_at[d] if c.is_true else 0
                avoid[i, d] = cfg.avoid_steps
                c.avoiders.append((i, detour, wait_avoided))
    return None


def reveal_checks(t, board, agents_xy, doorways, avoid, trust, cfg) -> None:
    """Reveal the truth of claims, by audit or by a passer-by who did not believe them.

    Inputs
        board       list[Claim] (modified in place)
        agents_xy   list of (y, x), padded coords
        doorways    list of (y, x) doorway cells, padded coords
        avoid       int array (N, D); avoid[j, d] > 0 means j currently believes d is jammed
        trust       float array (N, N) (modified in place)
    """
    N = len(agents_xy)
    for c in board:
        if c.checked or t - c.t > cfg.claim_ttl:
            continue
        if c.audited:
            c.checked = True
            trust[:, c.sender] += cfg.trust_lr * (float(c.is_true) - trust[:, c.sender])
            continue
        for i in range(N):
            if i == c.sender:
                continue
            if (
                avoid[i, c.doorway] > 0
            ):  # believer: routes around the doorway, never sees it
                continue
            if (
                chebyshev_dist(agents_xy[i], doorways[c.doorway]) > cfg.obs_radius
            ):  # too far to see it
                continue
            c.checked = True
            trust[i, c.sender] += cfg.trust_lr * (float(c.is_true) - trust[i, c.sender])
    return None


# ---------------------------------------------------------------- step 8: credit


def claim_credit(claim: Claim, cfg, rng) -> float:
    """Credit to claim.sender for one claim, under cfg.rule. Units: steps (mean value v_bar = 1 step).

    Per avoider (j, detour, wait_avoided) recorded on the claim, steps saved = wait_avoided - detour.

    none        return 0.0
    popularity  return cfg.beta * len(claim.avoiders)              (paid whether true or not)
    accuracy    return cfg.gamma * sum(wait - detour for _, detour, wait in claim.avoiders)
                (negative for a fake jam: avoiders only pay the detour)
    tvc         TVC's three parts, in step units (v_bar = 1); a claim is "true" or a factor-2 lie:
                  gain  = sum(min(max(wait - detour, 0), budget)) over avoiders, only if checked and true
                  bonus = eta * S, only if checked; S = 1 if true else exp(-(log 2)^2 / (2 w^2)) (~0)
                  fine  = fine, only if audited and false   (audit drawn at posting, independent of the claim)
                Same formula as tvc.tvc_credit (test_claim_credit checks they agree); written out directly
                because a claim revealed by a passer-by can be checked with no avoiders.
    """
    if cfg.rule == "none":
        return 0.0
    if cfg.rule == "popularity":
        return cfg.beta * len(claim.avoiders)
    if cfg.rule == "accuracy":
        return cfg.gamma * sum(wait - detour for _, detour, wait in claim.avoiders)
    if cfg.rule in (
        "fine",
        "pop_fine",
    ):  # lying cost alone, or popularity credit plus the same lying cost
        fined = cfg.tvc_params.fine * (claim.audited and not claim.is_true)
        return (
            cfg.beta * len(claim.avoiders) if cfg.rule == "pop_fine" else 0.0
        ) - fined
    # tvc
    params = cfg.tvc_params
    gain = sum(
        min(max(wait - detour, 0), params.budget) for _, detour, wait in claim.avoiders
    )
    score = (
        1.0
        if claim.is_true
        else np.exp(-(np.log(2.0) ** 2) / (2 * max(params.sigma, 0.05) ** 2))
    )
    # verification gate: a lie here is a factor-2 claim, |m / theta - 1| = 1, so it passes only if tau >= 1
    verified = claim.is_true or params.tau >= 1.0
    credit = 0.0
    if claim.checked:
        credit += gain * verified + params.eta * score
    if claim.audited and not verified:
        credit -= params.fine
    return float(credit)


# ---------------------------------------------------------------- episode loop + learning


def run_episode(
    cfg, rng, pol, trust, learn=True, forced=None, render=None, record=None
) -> dict[str, Any]:
    """One episode. Returns dict of metrics: throughput, fake_rate, true_report_rate, misled, detour, spend.
    render: optional .svg path; saves POGEMA's animation of the episode.
    record: optional dict; filled with walls, doorways and one frame per step (for make_video)."""
    map_rows, doorways = load_map(cfg.env, cfg)
    env = make_env(cfg, map_rows, int(rng.integers(1 << 30)))
    if render:
        from pogema import AnimationConfig, AnimationMonitor

        env = AnimationMonitor(
            env, AnimationConfig(save_every_idx_episode=None)
        )  # no auto-save to renders/
    obs, info = env.reset()
    walls = np.asarray(env.unwrapped.get_obstacles(), bool)
    zones = doorway_zones(doorways, walls, cfg.jam_radius)
    avoid = np.zeros((cfg.agents, len(doorways)), int)
    board, decisions, throughput = [], [], np.zeros(cfg.agents)
    prev_visible = [set() for _ in range(cfg.agents)]
    N, T = cfg.agents, cfg.steps
    posts = np.zeros(N)
    for t in range(T):
        agents_xy = [tuple(p) for p in env.unwrapped.get_agents_xy()]
        targets_xy = [tuple(p) for p in env.unwrapped.get_targets_xy()]
        active = (
            env.unwrapped.grid.is_active
        )  # one-shot: agents that reached their goal leave the map
        on_map = [p for i, p in enumerate(agents_xy) if active[i]]
        truth = np.array([jammed(z, on_map, cfg) for z in zones])
        queue_at = np.array([sum(p in z for p in on_map) for z in zones])

        # 1-2. plan: each agent's distance map under its own beliefs, its route, and what it can see
        my_route_doors, visible_doors, dists, blocked = [], [], [], []
        for j in range(N):
            blocked_j = walls.copy()
            for d in np.flatnonzero(avoid[j] > 0):
                blocked_j[doorways[d]] = True
            dist_j = bfs_dist(blocked_j, targets_xy[j])
            my_route_doors.append(set(route_doorways(dist_j, agents_xy[j], doorways)))
            visible_doors.append(
                [
                    d
                    for d, door in enumerate(doorways)
                    if chebyshev_dist(agents_xy[j], door) <= cfg.obs_radius
                ]
            )
            dists.append(dist_j)
            blocked.append(blocked_j)
        dists_through = np.array([dists[j][agents_xy[j]] for j in range(N)])

        def dists_around(j, d):  # distance to target if doorway d is also blocked
            b = blocked[j].copy()
            b[doorways[d]] = True
            return bfs_dist(b, targets_xy[j])[agents_xy[j]]

        # 3-5. talk, believe, check
        n_before = len(board)
        new = post_claims(
            t,
            visible_doors,
            my_route_doors,
            truth,
            pol,
            cfg,
            rng,
            board,
            forced,
            prev_visible,
        )
        posted = iter(
            board[n_before:]
        )  # post_claims appends one claim per action 1, in decision order
        decisions += [(i, s, a, p, next(posted) if a else None) for i, s, a, p in new]
        for c in board[n_before:]:
            posts[c.sender] += 1
        adopt_claims(
            t,
            board,
            agents_xy,
            avoid,
            trust,
            dists_through,
            dists_around,
            my_route_doors,
            queue_at,
            cfg,
            rng,
        )
        reveal_checks(t, board, agents_xy, doorways, avoid, trust, cfg)

        # 6. move (believers re-plan around avoided doorways from next step)
        actions = [next_action(dists[j], agents_xy[j], rng) for j in range(N)]
        obs, rew, term, trunc, info = env.step(actions)
        throughput += np.asarray(rew, float)
        if record is not None:
            record.setdefault("walls", walls)
            record.setdefault("doorways", doorways)
            record.setdefault("frames", []).append(
                dict(
                    agents=agents_xy,
                    targets=targets_xy,
                    jammed=truth.copy(),
                    claims=[
                        (c.doorway, c.is_true)
                        for c in board
                        if t - c.t <= cfg.claim_ttl
                    ],
                    believers=int((avoid > 0).sum()),
                    delivered=float(throughput.sum()),
                )
            )
        avoid = np.maximum(avoid - 1, 0)
        if all(
            np.logical_or(term, trunc)
        ):  # one-shot: everyone arrived (or time limit)
            break

    if render:
        Path(render).parent.mkdir(parents=True, exist_ok=True)
        env.save_animation(str(render))  # writes to this exact path

    credit_of = {id(c): claim_credit(c, cfg, rng) for c in board}
    credit = np.zeros(N)
    for c in board:
        credit[c.sender] += credit_of[id(c)]
    lam = cfg.prosocial
    others = (throughput.sum() - throughput) / (N - 1)
    social = (1 - lam) * throughput + lam * others
    reward = social + credit - cfg.report_cost * posts

    if learn:
        # REINFORCE, per decision, with the claim's own effect only (silence = 0):
        #   (1-lam) * own gain  +  lam * effect on the others' mean  +  credit  -  report cost
        # Own gain and effect are per believer (cfg.own_gain / harm / benefit, calibrated in deliveries), so the
        # natural temptation to lie (G_1) and the prosocial weight lam both reach each choice.
        # Baseline per (agent, state).
        batch = []  # (robot, state, action, probs, reward) for learners with their own update
        for i, state, action, probs, claim in decisions:
            r = 0.0
            if claim:
                n_bel = len(claim.avoiders)
                own = cfg.own_gain * n_bel if state >= 2 else 0.0  # claim on my route
                ext = (cfg.benefit if claim.is_true else -cfg.harm) * n_bel
                r = (1 - lam) * own + lam * ext + credit_of[id(claim)] - cfg.report_cost
            if hasattr(pol, "update"):
                batch.append((i, state, action, probs, r))
                continue
            adv = np.clip(r - pol.base[i, state], -10, 10)
            pol.logits[i, state] += cfg.lr * adv * (np.eye(2)[action] - probs)
            pol.base[i, state] += 0.05 * (r - pol.base[i, state])
        if batch:
            pol.update(batch)

    avoiders = [(c, a) for c in board for a in c.avoiders]
    in_state = lambda s: [d[2] for d in decisions if d[1] == s]
    rate = lambda xs: float(sum(xs) / max(len(xs), 1))
    return dict(
        throughput=float(
            throughput.sum() / (N * T)
        ),  # deliveries per agent-step (welfare)
        fake_rate=float(sum(not c.is_true for c in board) / max(len(board), 1)),
        lie_rate=rate(
            in_state(2)
        ),  # P(post | my route, free): the selfish lie, free of base rates
        useful_report_rate=rate(in_state(1)),  # P(post | elsewhere, jammed)
        misled=float(sum(not c.is_true for c, _ in avoiders) / max(len(avoiders), 1)),
        detour=float(sum(a[1] for _, a in avoiders) / N),
        spend=float(np.maximum(credit, 0).sum() / (N * T)),
        audits=float(
            sum(c.audited for c in board) / T
        ),  # audited claims per step, the cost of verification
        reward=float(reward.mean()),
        throughput_per_agent=throughput.copy(),
        posts_per_agent=posts.copy(),
        **{f"pogema_{k}": float(v) for k, v in info[0].get("metrics", {}).items()},
        believers_per_agent=np.bincount(
            [c.sender for c in board for _ in c.avoiders], minlength=N
        ),
    )


HAND_SET = {  # robot 0 only; everyone else silent
    "silent": lambda i, s: 0,
    "reporter": lambda i, s: int(i == 0 and s == 1),  # reports real jams elsewhere
    "liar": lambda i, s: int(i == 0 and s == 2),  # fake jams on its own route
    "all_liars": lambda i, s: int(
        s == 2
    ),  # everyone hypes: where the collective harm shows
}


def _hand_set_episode(
    args,
) -> dict[str, Any]:  # top-level so the process pool can pickle it
    cfg, name, e = args
    return run_episode(
        cfg,
        np.random.default_rng(1000 + e),
        make_policy(cfg),
        np.ones((cfg.agents, cfg.agents)),
        learn=False,
        forced=HAND_SET[name],
    )


def measure_B_G_H(cfg, episodes=50) -> dict[str, float]:
    """Hand-set worlds, rule 'none', common seeds per episode. Per episode (deliveries):
    B (benefit)  others(reporter) − others(silent)   how much honest reports help everyone else
    G (gain)     robot 0(liar) − robot 0(silent)     how much lying helps the liar itself
    H (harm)     others(silent) − others(liar)       how much lying hurts everyone else
    Per believer of robot 0's claims: own_gain = G / believers(liar), harm = H / believers(liar),
    benefit = B / believers(reporter). Also lies per episode and believers per lie."""
    cfg = replace(cfg, rule="none")
    jobs = [(cfg, name, e) for name in HAND_SET for e in range(episodes)]
    with ProgressPool() as pool:
        runs = list(pool.map(_hand_set_episode, jobs))
    m = {}
    for k, name in enumerate(HAND_SET):
        rs = runs[k * episodes : (k + 1) * episodes]
        tp = np.mean([r["throughput_per_agent"] for r in rs], axis=0)
        m[name] = dict(
            own0=tp[0],
            others=tp[1:].mean(),
            all=tp.mean(),
            posts0=np.mean([r["posts_per_agent"][0] for r in rs]),
            bel0=np.mean([r["believers_per_agent"][0] for r in rs]),
            bel_all=np.mean([r["believers_per_agent"].sum() for r in rs]),
        )
    N = cfg.agents
    B = m["reporter"]["others"] - m["silent"]["others"]
    G = m["liar"]["own0"] - m["silent"]["own0"]
    H = (
        m["silent"]["others"] - m["liar"]["others"]
    )  # one liar among silent robots (can be < 0)
    bel_lie, bel_rep = max(m["liar"]["bel0"], 1e-9), max(m["reporter"]["bel0"], 1e-9)
    own_gain = G / bel_lie
    # harm per believer measured where everyone hypes: total loss plus the liars' own gains = what believers
    # lost, spread over the other N-1 robots' mean. A single liar among silent robots barely hurts (traffic
    # spreads, price-of-anarchy effect); the harm is collective.
    bel_all = max(m["all_liars"]["bel_all"], 1e-9)
    harm = (
        (N * (m["silent"]["all"] - m["all_liars"]["all"]) + own_gain * bel_all)
        / bel_all
        / (N - 1)
    )
    return dict(
        B=float(B),
        G=float(G),
        H=float(H),
        own_gain=float(own_gain),
        harm=float(harm),
        benefit=float(B / bel_rep),
        lies=float(m["liar"]["posts0"]),
        believers_per_lie=float(bel_lie / max(m["liar"]["posts0"], 1e-9)),
    )


def calibrate(cfg, episodes=30) -> Config:
    """cfg with own_gain / harm / benefit set from measure_B_G_H; prints the fine lower bound phi_lo."""
    m = measure_B_G_H(cfg, episodes)
    lam, c, p = cfg.prosocial, cfg.report_cost, cfg.tvc_params.p_audit
    per_lie = m["believers_per_lie"] * ((1 - lam) * m["own_gain"] - lam * m["harm"])
    print(
        "calibration: "
        + "  ".join(f"{k} {v:.4f}" for k, v in m.items())
        + f"\n  net private gain per lie at lam={lam}: {per_lie - c:.4f}"
        + f"   phi_lo = {(per_lie - c) / p:.4f}   lie pays without credit iff lam < "
        + f"{m['own_gain'] / (m['own_gain'] + m['harm']):.3f} (ignoring c)",
        flush=True,
    )
    return replace(cfg, own_gain=m["own_gain"], harm=m["harm"], benefit=m["benefit"])


def train(cfg) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    pol, trust = make_policy(cfg), np.ones((cfg.agents, cfg.agents))
    return [run_episode(cfg, rng, pol, trust) for _ in range(cfg.episodes)]


# ---------------------------------------------------------------- training summaries (top-level: process pools pickle them)


SHOW = ("lie_rate", "useful_report_rate", "throughput", "spend")


def _train_all_metrics(
    cfg,
) -> dict[str, float]:  # every scalar metric, incl. POGEMA's; top-level for the pool
    hist = train(cfg)
    return summarize(hist, [k for k, v in hist[-1].items() if isinstance(v, float)])


def _train_summary(
    cfg,
) -> dict[str, float]:  # top-level so the process pool can pickle it
    return summarize(train(cfg), SHOW)


# ---------------------------------------------------------------- figures and video

LOGS = Path(__file__).resolve().parents[2] / "logs"


def _hype_episode(
    args,
) -> dict[str, Any]:  # top-level so the process pool can pickle it
    cfg, k, seed = args
    forced = lambda i, s: int(s in (2, 3)) if i < k else int(s in (1, 3))
    m = run_episode(
        cfg,
        np.random.default_rng(seed),
        make_policy(cfg),
        np.ones((cfg.agents, cfg.agents)),
        learn=False,
        forced=forced,
    )
    per = m["throughput_per_agent"]  # deliveries per robot per episode
    nan = float("nan")
    return dict(
        group=per.mean(),
        hypers=per[:k].mean() if k else nan,
        honest=per[k:].mean() if k < cfg.agents else nan,
        misled=m["misled"],
    )


# ---------------------------------------------------------------- experiments


def hype_sweep(
    cfg, episodes=20, seeds=10, ks=(0, 1, 2, 3, 6, 9, 11, 12)
):  # receiver type = cfg.receiver; returns (rows, file tag)
    """Hand-set, no learning, rule 'none': robots i < k hype (post fake jams on their own route),
    the rest report real jams honestly. Same episode seeds for every k, so differences come from hype alone.
    `seeds` independent batches of `episodes` episodes; the CI is over the batch means."""
    cfg = replace(cfg, rule="none")
    tag = (
        "" if cfg.receiver == "naive" else "_" + cfg.receiver
    )  # keep the naive-receiver files
    jobs = [
        (cfg, k, 1000 + b * episodes + e)
        for k in ks
        for b in range(seeds)
        for e in range(episodes)
    ]
    with ProgressPool() as pool:
        res = list(pool.map(_hype_episode, jobs, chunksize=4))
    per_k = episodes * seeds
    rows, lines = (
        [],
        [
            f"### Hype sweep (hand-set, no credit, {cfg.receiver} receivers)\n",
            f"{seeds} seeds x {episodes} episodes per k, 95% CI over seeds\n",
            "| k hype robots | group | honest | hypers | misled |",
            "| --- | --- | --- | --- | --- |",
        ],
    )
    for n, k in enumerate(ks):
        eps = res[n * per_k : (n + 1) * per_k]
        row = {"k": k}
        for key in ("group", "honest", "hypers", "misled"):
            batches = [
                np.nanmean([e[key] for e in eps[b * episodes : (b + 1) * episodes]])
                if not np.isnan(
                    [e[key] for e in eps[b * episodes : (b + 1) * episodes]]
                ).all()
                else np.nan
                for b in range(seeds)
            ]
            row[key] = (
                ci95(batches) if not np.isnan(batches).any() else (np.nan, np.nan)
            )
        rows.append(row)
        lines.append(
            f"| {k} | "
            + " | ".join(
                "{:.3f} ± {:.3f}".format(*row[key])
                for key in ("group", "honest", "hypers", "misled")
            )
            + " |"
        )
        print(lines[-1], flush=True)
    LOGS.mkdir(exist_ok=True)
    (LOGS / f"fake_jams_hype{tag}.md").write_text("\n".join(lines) + "\n")
    return rows, tag


def sweep(
    make_cfg, values, rules, episodes, seeds=1, table=None, label="value"
) -> dict[str, list[dict[str, float]]]:
    """For each rule and value: train `seeds` runs of `episodes` episodes, return end-of-training metrics
    (mean over seeds, plus `<metric>_ci` = 95% CI half-width). Runs are independent, so they go to a process pool.
    table: optional .md path; writes one row per (rule, value) as mean ± CI."""
    cfgs = [
        replace(make_cfg(v), rule=rule, episodes=episodes, seed=s)
        for rule in rules
        for v in values
        for s in range(seeds)
    ]
    with ProgressPool() as pool:
        results = iter(pool.map(_train_summary, cfgs))
    out = {rule: [] for rule in rules}
    for rule in rules:
        for v in values:
            runs = [next(results) for _ in range(seeds)]
            row = {}
            for k in SHOW:
                row[k], row[k + "_ci"] = ci95([r[k] for r in runs])
            out[rule].append(row)
            print(
                f"  {rule:<10} {v:<6} "
                + "  ".join(f"{k} {row[k]:.3f}±{row[k + '_ci']:.3f}" for k in SHOW),
                flush=True,
            )
    if table:
        lines = [
            f"{seeds} seeds, {episodes} episodes, last-50 mean ± 95% CI\n",
            f"| rule | {label} | " + " | ".join(SHOW) + " |",
            "| --- " * (len(SHOW) + 2) + "|",
        ]
        for rule in rules:
            for v, row in zip(values, out[rule]):
                lines.append(
                    f"| {rule} | {v} | "
                    + " | ".join(f"{row[k]:.3f} ± {row[k + '_ci']:.3f}" for k in SHOW)
                    + " |"
                )
        Path(table).parent.mkdir(exist_ok=True)
        Path(table).write_text("\n".join(lines) + "\n")
        print(f"-> {table}", flush=True)
    return out


def run_variants(
    variants, episodes, seeds, table, title
) -> dict[str, dict[str, tuple[float, float]]]:
    """Train each named config for `seeds` seeds in a process pool; write a mean ± 95% CI table (SHOW metrics)."""
    names = list(variants)
    cfgs = [
        replace(variants[n], episodes=episodes, seed=s)
        for n in names
        for s in range(seeds)
    ]
    with ProgressPool() as pool:
        res = list(pool.map(_train_summary, cfgs))
    lines = [
        f"### {title}\n",
        f"{seeds} seeds, {episodes} episodes, last-50 mean ± 95% CI\n",
        "| variant | " + " | ".join(SHOW) + " |",
        "| --- " * (len(SHOW) + 1) + "|",
    ]
    out = {}
    for k, n in enumerate(names):
        runs = res[k * seeds : (k + 1) * seeds]
        out[n] = {m: ci95([r[m] for r in runs]) for m in SHOW}
        lines.append(
            f"| {n} | "
            + " | ".join("{:.3f} ± {:.3f}".format(*out[n][m]) for m in SHOW)
            + " |"
        )
        print(
            f"  {n:<32} "
            + "  ".join("{} {:.3f}±{:.3f}".format(m, *out[n][m]) for m in SHOW),
            flush=True,
        )
    LOGS.mkdir(exist_ok=True)
    Path(table).write_text("\n".join(lines) + "\n")
    print(f"-> {table}", flush=True)
    return out


if __name__ == "__main__":
    # python -m tvc.fake_jams hype [episodes]           hand-set: more hype robots -> lower group deliveries
    # python -m tvc.fake_jams bgh  [episodes]           measure B, G, H; print lam window and TVC fine threshold
    # python -m tvc.fake_jams lam  [episodes] [seeds]   headline: rules x prosocial weight lam
    # python -m tvc.fake_jams fine [episodes] [seeds]   TVC fine sweep around the theory threshold
    # python -m tvc.fake_jams audit [episodes] [seeds]  TVC audit-rate sweep: fixed fine vs fine = 2 * phi_lo(p)
    import sys
    from tvc.visualisation import (
        OUT,
        make_video,
        mean_ci,
        plot_hype,
        plot_audit,
        plot_hype_compare,
        plot_listen,
        plot_scale,
        plot_sweep,
        read_table,
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "bgh"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (20 if mode == "bgh" else 150)
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    base = Config()

    if mode == "bgh":
        calibrate(base, n)
    elif mode == "hype":
        rows, tag = hype_sweep(
            replace(base, receiver=sys.argv[4]) if len(sys.argv) > 4 else base, n, seeds
        )
        plot_hype(rows, tag)
    elif mode == "hype_compare":  # one figure from the naive and sceptical hype tables
        plot_hype_compare()
    elif mode == "scale":
        # scaling: agents and map size, selfish robots (lam = 0), each layout calibrated on its own
        layouts = [
            (3, 8),
            (3, 12),
            (3, 16),
            (3, 32),
            (4, 16),
            (4, 32),
        ]  # (rooms per side, agents)
        rules = ("none", "popularity", "tvc")
        keys = ("lie_rate", "useful_report_rate", "throughput", "pogema_avg_throughput")
        rows = {}
        for rooms, agents in layouts:
            cfg_l = calibrate(
                replace(
                    base,
                    env="rooms" if rooms == 3 else "rooms4",
                    agents=agents,
                    prosocial=0.0,
                )
            )
            cfgs = [
                replace(cfg_l, rule=r, episodes=n, seed=s)
                for r in rules
                for s in range(seeds)
            ]
            with ProgressPool() as pool:
                res = list(pool.map(_train_all_metrics, cfgs))
            for k_r, r in enumerate(rules):
                runs = res[k_r * seeds : (k_r + 1) * seeds]
                rows[(rooms, agents, r)] = {k: ci95([x[k] for x in runs]) for k in keys}
                print(
                    f"  {rooms}x{rooms} N={agents:<3} {r:<10} "
                    + "  ".join(
                        f"{k} {m:.3f}±{h:.3f}"
                        for k, (m, h) in rows[(rooms, agents, r)].items()
                    ),
                    flush=True,
                )
        lines = [
            f"{seeds} seeds, {n} episodes, lam = 0, last-50 mean ± 95% CI\n",
            "| rooms | agents | rule | " + " | ".join(keys) + " |",
            "| --- " * (len(keys) + 3) + "|",
        ]
        for (rooms, agents, r), v in rows.items():
            lines.append(
                f"| {rooms}x{rooms} | {agents} | {r} | "
                + " | ".join(f"{m:.3f} ± {h:.3f}" for m, h in v.values())
                + " |"
            )
        LOGS.mkdir(exist_ok=True)
        (LOGS / "fake_jams_scale.md").write_text("\n".join(lines) + "\n")
        print("-> logs/fake_jams_scale.md")

        plot_scale(rows, layouts, rules, OUT / "fake_jams_scale.png")
        print("-> figures/fake_jams_scale.png")
    elif mode == "pogema":
        # POGEMA's own metrics, per credit rule, selfish robots (lam = 0), both POGEMA tasks
        lines = []
        for task in ("restart",):
            cfg_t = calibrate(replace(base, on_target=task, prosocial=0.0))
            lam = (
                cfg_t.agents - 1
            ) / cfg_t.agents  # shared return: every robot gets the mean deliveries
            designs = {
                "none": replace(cfg_t, rule="none"),
                "popularity": replace(cfg_t, rule="popularity"),
                "accuracy": replace(cfg_t, rule="accuracy"),
                "fine": replace(cfg_t, rule="fine"),
                "pop_fine": replace(cfg_t, rule="pop_fine"),
                "reputation": replace(cfg_t, rule="none", receiver="sceptical"),
                "shared": replace(cfg_t, rule="none", prosocial=lam),
                "tvc": replace(cfg_t, rule="tvc"),
            }
            rules = tuple(designs)
            cfgs = [
                replace(designs[r], episodes=n, seed=s)
                for r in rules
                for s in range(seeds)
            ]
            with ProgressPool() as pool:
                res = list(pool.map(_train_all_metrics, cfgs))
            keys = ["lie_rate", "useful_report_rate"] + sorted(
                k for k in res[0] if k.startswith("pogema_")
            )
            lines += [
                f"\n### POGEMA task: {task} (lam = 0, {seeds} seeds, {n} episodes, last-50 mean ± 95% CI over seeds)\n",
                "| rule | "
                + " | ".join(k.removeprefix("pogema_") for k in keys)
                + " |",
                "| --- " * (len(keys) + 1) + "|",
            ]
            for k_r, r in enumerate(rules):
                runs = res[k_r * seeds : (k_r + 1) * seeds]
                cells = [
                    "{:.3f} ± {:.3f}".format(*ci95([x[k] for x in runs])) for k in keys
                ]
                lines.append(f"| {r} | " + " | ".join(cells) + " |")
            print("\n".join(lines[-(len(rules) + 3) :]), flush=True)
        LOGS.mkdir(exist_ok=True)
        (LOGS / "fake_jams_pogema.md").write_text("\n".join(lines) + "\n")
        print("-> logs/fake_jams_pogema.md")
    elif mode == "ablate":
        # baselines and TVC ablations at lam = 0 (most selfish): which parts are needed?
        b = calibrate(replace(base, prosocial=0.0))
        p = b.tvc_params
        variants = {
            "none": replace(b, rule="none"),
            "popularity": replace(b, rule="popularity"),
            "accuracy": replace(b, rule="accuracy"),
            f"TVC (fine {p.fine:g}, default)": replace(b, rule="tvc"),
            "TVC fine 8": replace(b, rule="tvc", tvc_params=replace(p, fine=8.0)),
            "TVC no bonus (eta 0)": replace(
                b, rule="tvc", tvc_params=replace(p, eta=0.0)
            ),
            "TVC no audit (p 0)": replace(
                b, rule="tvc", tvc_params=replace(p, p_audit=0.0)
            ),
            "TVC no fine (fine 0)": replace(
                b, rule="tvc", tvc_params=replace(p, fine=0.0)
            ),
            # no gate: every claim passes verification, so lies keep the gain credit and are never fined
            "TVC no gate (tau 1e6)": replace(
                b, rule="tvc", tvc_params=replace(p, tau=1e6)
            ),
            "TVC no cap (budget 1e6)": replace(
                b, rule="tvc", tvc_params=replace(p, budget=1e6)
            ),
        }
        sel = (
            sys.argv[4] if len(sys.argv) > 4 else None
        )  # e.g. "gate": rerun only the variants with that word
        if sel:
            variants = {
                k: v
                for k, v in variants.items()
                if sel in k or k.startswith("TVC (fine")
            }
        run_variants(
            variants,
            n,
            seeds,
            LOGS / (f"fake_jams_ablate_{sel}.md" if sel else "fake_jams_ablate.md"),
            "Baselines and TVC ablations (lam = 0)",
        )
    elif mode == "sceptical":
        # receivers that learn per-sender trust from checks (reputation): is reputation alone enough?
        b = calibrate(replace(base, receiver="sceptical", prosocial=0.0))
        run_variants(
            {
                f"{r}, lam {lam:g}": replace(b, rule=r, prosocial=lam)
                for lam in (0.0, 0.5)
                for r in ("none", "popularity", "tvc")
            },
            n,
            seeds,
            LOGS / "fake_jams_sceptical.md",
            "Sceptical (trust-learning) receivers",
        )
    elif mode == "shared":
        # shared return: every robot is paid the mean deliveries of all robots, own included.
        # In r = (1 - lam) * own + lam * others-mean that is lam = (N - 1) / N.
        b = calibrate(replace(base, prosocial=0.0))
        lam = (b.agents - 1) / b.agents
        run_variants(
            {
                f"{r}, lam {lam:.3f}": replace(b, rule=r, prosocial=lam)
                for r in ("none", "tvc")
            },
            n,
            seeds,
            LOGS / "fake_jams_shared.md",
            f"Shared return (lam = (N-1)/N = {lam:.3f})",
        )
    elif mode == "envs":
        # credit rules, reputation and shared return on each environment (python -m tvc.fake_jams envs 150 10 rooms rooms4)
        variants = {}
        for env in sys.argv[4:] or ("rooms", "rooms4"):
            b = calibrate(replace(base, env=env, prosocial=0.0))
            lam = (
                (b.agents - 1) / b.agents
            )  # shared return: every robot is paid the mean deliveries of all robots
            variants |= {
                f"{env}: {r}": replace(b, rule=r)
                for r in ("none", "popularity", "accuracy", "tvc")
            }
            variants[f"{env}: reputation (sceptical)"] = replace(
                b, rule="none", receiver="sceptical"
            )
            variants[f"{env}: shared return (lam {lam:.3f})"] = replace(
                b, rule="none", prosocial=lam
            )
        run_variants(
            variants,
            n,
            seeds,
            LOGS / "fake_jams_envs.md",
            "Environments (lam = 0 unless noted)",
        )
    elif mode == "listen":
        # rho = P(receiver acts on a claim): does TVC raise welfare only where information changes behaviour?
        # calibrated once at rho = 1; python -m tvc.fake_jams listen 150 10 rooms
        rhos, envs, variants = (
            (0.0, 0.25, 0.5, 0.75, 1.0),
            sys.argv[4:] or ["rooms"],
            {},
        )
        for env in envs:
            b = calibrate(replace(base, env=env, prosocial=0.0))
            variants |= {
                f"{env}, rho {rho:g}: {r}": replace(b, rule=r, listen=rho)
                for rho in rhos
                for r in ("none", "tvc")
            }
        res = run_variants(
            variants, n, seeds, LOGS / "fake_jams_listen.md", "Deaf receivers (lam = 0)"
        )
        plot_listen(res, envs, rhos, OUT / "fake_jams_listen.png")
        print("-> figures/fake_jams_listen.png")
    elif mode == "sensitivity":
        # scale the calibrated private gain; phi_lo should move with it (lam = 0, so harm has no weight)
        b = calibrate(replace(base, prosocial=0.0))
        bel = measure_B_G_H(b, 10)["believers_per_lie"]
        p = b.tvc_params
        variants = {}
        for k in (0.5, 1.0, 2.0):
            bk = replace(b, own_gain=b.own_gain * k, harm=b.harm * k)
            phi = max((bel * bk.own_gain - bk.report_cost) / p.p_audit, 0.0)
            variants[f"gain x{k:g}: none"] = replace(bk, rule="none")
            variants[f"gain x{k:g}: TVC fine {phi:.2f} (phi_lo)"] = replace(
                bk, rule="tvc", tvc_params=replace(p, fine=phi)
            )
            variants[f"gain x{k:g}: TVC fine {2 * phi:.2f} (2 phi_lo)"] = replace(
                bk, rule="tvc", tvc_params=replace(p, fine=2 * phi)
            )
            variants[f"gain x{k:g}: TVC fine {p.fine:g} (default)"] = replace(
                bk, rule="tvc"
            )
        run_variants(
            variants,
            n,
            seeds,
            LOGS / "fake_jams_sensitivity.md",
            "Calibration sensitivity (lam = 0)",
        )
    elif mode == "replot":
        # redraw figures from the saved tables (logs/*.md), no training
        for name, label in (
            ("lam", "prosocial weight lam"),
            ("fine", "TVC fine (lam = 0)"),
        ):
            rows = read_table(LOGS / f"fake_jams_{name}.md")
            vcol = [k for k in rows[0] if k not in SHOW and k != "rule"][0]
            values = sorted({float(r[vcol]) for r in rows})
            out = {}
            for r in rows:
                row = {}
                for k in SHOW:
                    row[k], row[k + "_ci"] = mean_ci(r[k])
                out.setdefault(r["rule"], []).append(row)
            plot_sweep(out, values, label, OUT / f"fake_jams_{name}.png")
        rows = read_table(LOGS / "fake_jams_scale.md")
        keys = ("lie_rate", "useful_report_rate", "throughput", "pogema_avg_throughput")
        table = {
            (int(r["rooms"][0]), int(r["agents"]), r["rule"]): {
                k: mean_ci(r[k]) for k in keys
            }
            for r in rows
        }
        layouts = sorted({(k[0], k[1]) for k in table})
        plot_scale(
            table, layouts, ("none", "popularity", "tvc"), OUT / "fake_jams_scale.png"
        )
        print("-> figures/fake_jams_{lam,fine,scale}.png")
    elif mode == "video":
        make_video(base, OUT / "fake_jams_honest_vs_hype.mp4")
        print("-> figures/fake_jams_honest_vs_hype.mp4")
    elif mode == "anim":
        # same seed, two worlds: everyone reports real jams vs everyone hypes. Open the .svg files in a browser.
        for name, forced in (
            ("honest", lambda i, s: int(s in (1, 3))),
            ("hype", lambda i, s: int(s == 2)),
        ):
            m = run_episode(
                replace(base, rule="none"),
                np.random.default_rng(7),
                make_policy(base),
                np.ones((base.agents, base.agents)),
                learn=False,
                forced=forced,
                render=OUT / f"fake_jams_{name}.svg",
            )
            print(
                f"{name}: {m['throughput'] * base.steps:.2f} deliveries per robot -> figures/fake_jams_{name}.svg"
            )
    elif mode == "lam":
        base = calibrate(base)
        lams = [0.0, 0.25, 0.5, 0.75, 1.0]
        out = sweep(
            lambda v: replace(base, prosocial=v),
            lams,
            ("none", "popularity", "tvc"),
            n,
            seeds,
            table=LOGS / "fake_jams_lam.md",
            label="lam",
        )
        plot_sweep(out, lams, "prosocial weight lam", OUT / "fake_jams_lam.png")
    elif mode == "fine":
        # selfish robots (lam = 0): the strongest temptation; fines bracket the predicted threshold phi_lo
        base = calibrate(replace(base, prosocial=0.0))
        m_lie = base.own_gain  # per believer
        bel = measure_B_G_H(base, 10)["believers_per_lie"]
        phi_lo = (bel * m_lie - base.report_cost) / base.tvc_params.p_audit
        print(f"fine sweep around phi_lo = {phi_lo:.4f}", flush=True)
        fines = sorted(
            {0.0, *(round(k * max(phi_lo, 0.05), 4) for k in (0.25, 0.5, 1, 2, 4, 8))}
        )
        out = sweep(
            lambda v: replace(base, tvc_params=replace(base.tvc_params, fine=v)),
            fines,
            ("tvc",),
            n,
            seeds,
            table=LOGS / "fake_jams_fine.md",
            label=f"fine (phi_lo = {phi_lo:.2f})",
        )
        plot_sweep(out, fines, "TVC fine (lam = 0)", OUT / "fake_jams_fine.png")
    elif mode == "audit":
        # selfish robots (lam = 0). Lying stops when p * fine > gap, where gap = (net private gain of a lie) = Gamma - xi.
        # Arm A keeps the fine at 4 * phi_lo(0.2) and lowers p; arm B sets fine = 2 * gap / p at every p.
        b = calibrate(replace(base, prosocial=0.0))
        gap = measure_B_G_H(b, 10)["believers_per_lie"] * b.own_gain - b.report_cost
        fixed = 4 * gap / 0.2
        ps = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
        arms = {
            "fixed fine": lambda p: fixed,
            "fine = 2 phi_lo(p)": lambda p: 2 * gap / p,
        }
        print(
            f"gap {gap:.4f}  fixed fine {fixed:.3f}  predicted p* = {gap / fixed:.3f}",
            flush=True,
        )
        keys = ("lie_rate", "useful_report_rate", "pogema_avg_throughput", "audits")
        cfgs = [
            replace(
                b,
                rule="tvc",
                episodes=n,
                seed=s,
                tvc_params=replace(b.tvc_params, p_audit=p, fine=fine(p)),
            )
            for fine in arms.values()
            for p in ps
            for s in range(seeds)
        ]
        with ProgressPool() as pool:
            res = list(pool.map(_train_all_metrics, cfgs))
        rows, lines = (
            {},
            [
                f"gap = {gap:.4f}, fixed fine = {fixed:.3f}, predicted p* = {gap / fixed:.3f}; {seeds} seeds, {n} episodes, last-50 mean ± 95% CI\n",
                "| arm | p | fine | " + " | ".join(keys) + " |",
                "| --- " * (len(keys) + 3) + "|",
            ],
        )
        it = iter(res)
        for name, fine in arms.items():
            rows[name] = []
            for p in ps:
                runs = [next(it) for _ in range(seeds)]
                row = {k: ci95([x[k] for x in runs]) for k in keys}
                rows[name].append(row)
                lines.append(
                    f"| {name} | {p} | {fine(p):.3g} | "
                    + " | ".join("{:.3f} ± {:.3f}".format(*row[k]) for k in keys)
                    + " |"
                )
                print(lines[-1], flush=True)
        LOGS.mkdir(exist_ok=True)
        (LOGS / "fake_jams_audit.md").write_text("\n".join(lines) + "\n")
        print("-> logs/fake_jams_audit.md")
        plot_audit(rows, ps, gap / fixed, OUT / "fake_jams_audit.png")
        print("-> figures/fake_jams_audit.png")
    else:
        raise SystemExit(
            f"unknown mode {mode!r}; use one of: bgh hype scale audit pogema video anim lam fine replot ablate sceptical shared envs sensitivity"
        )
