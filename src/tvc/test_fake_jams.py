"""Checks for fake_jams.py, one per build step. Run: python test_fake_jams.py"""

import numpy as np

import tvc.fake_jams as fj
import tvc.pogema_env as pe
import tvc.util as util


def test_map_connected():
    cfg = fj.Config()
    rows, doorways = pe.make_rooms_map(cfg, np.random.default_rng(0))
    free = np.array([[ch == "." for ch in row] for row in rows])
    R = cfg.rooms
    assert len(doorways) == 2 * R * (
        R - 1
    )  # one doorway per pair of neighbouring rooms

    # flood fill: every free cell is reachable from the first one
    start = tuple(np.argwhere(free)[0])
    seen, stack = {start}, [start]
    while stack:
        y, x = stack.pop()
        for dy, dx in pe.MOVES[1:]:
            q = (y + dy, x + dx)
            if (
                0 <= q[0] < free.shape[0]
                and 0 <= q[1] < free.shape[1]
                and free[q]
                and q not in seen
            ):
                seen.add(q)
                stack.append(q)
    assert len(seen) == free.sum()

    # POGEMA accepts the map, and doorways (padded coords) are free in its obstacle grid
    env = pe.make_env(cfg, rows, seed=0)
    env.reset()
    obstacles = np.asarray(env.unwrapped.get_obstacles())
    assert all(obstacles[d] == 0 for d in doorways)


def test_bfs_and_planner():
    # empty grid: BFS distance equals Manhattan distance
    blocked = np.zeros((7, 9), bool)
    d = util.bfs_dist(blocked, (3, 4))
    yy, xx = np.indices(blocked.shape)
    assert (d == abs(yy - 3) + abs(xx - 4)).all()

    # wall with one gap: distance goes around it; cut-off cells are unreachable
    blocked[:, 2] = True
    blocked[0, 2] = False  # single gap at the top
    d = util.bfs_dist(blocked, (6, 0))
    assert (
        d[6, 4] == 6 + 2 + 6 + 2
    )  # up to row 0, through the gap, down to row 6, right to col 4
    blocked[0, 2] = True
    assert util.bfs_dist(blocked, (6, 0))[6, 4] == np.inf

    # following next_action from every reachable cell reaches the target in exactly dist steps
    blocked[0, 2] = False
    target = (6, 0)
    d = util.bfs_dist(blocked, target)
    rng = np.random.default_rng(0)
    for start in zip(*np.where(np.isfinite(d))):
        pos, steps = start, 0
        while pos != target:
            a = pe.next_action(d, pos, rng)
            assert a != 0
            pos = (pos[0] + pe.MOVES[a][0], pos[1] + pe.MOVES[a][1])
            steps += 1
        assert steps == d[start]
    assert pe.next_action(d, target) == 0  # at the target: stay
    blocked[0, 2] = True
    assert (
        pe.next_action(util.bfs_dist(blocked, target), (6, 4)) == 0
    )  # unreachable: stay


def test_jammed():
    cfg = fj.Config()
    rows, doorways = pe.make_rooms_map(cfg, np.random.default_rng(0))
    env = pe.make_env(cfg, rows, seed=0)
    env.reset()
    walls = np.asarray(env.unwrapped.get_obstacles(), bool)
    zones = pe.doorway_zones(doorways, walls, cfg.jam_radius)

    for (y, x), zone in zip(doorways, zones):
        assert all(not walls[c] for c in zone)  # zone holds only free cells
        assert all(
            abs(c[0] - y) + abs(c[1] - x) <= cfg.jam_radius for c in zone
        )  # within k steps

    (y, x), zone = doorways[0], zones[0]
    near = sorted(zone - {(y, x)})  # cells beside the doorway, reachable on foot
    assert pe.jammed(zone, near[: cfg.jam_queue], cfg)
    assert not pe.jammed(zone, near[: cfg.jam_queue - 1], cfg)

    # a cell within Chebyshev distance k but behind a wall is not in the zone, so agents there do not count
    behind = [
        (y + a, x + b)
        for a in range(-2, 3)
        for b in range(-2, 3)
        if not walls[y + a, x + b] and (y + a, x + b) not in zone
    ]
    assert (
        not pe.jammed(zone, near[: cfg.jam_queue - 1] + behind[:1], cfg) or not behind
    )


def test_post_claims():
    cfg = fj.Config(agents=3)
    pol, rng = fj.make_policy(cfg), np.random.default_rng(0)
    visible = [
        [0, 1],
        [1],
        [],
    ]  # agent 0 sees doors 0,1; agent 1 sees door 1; agent 2 none
    route = [{0}, set(), set()]  # door 0 is on agent 0's route
    truth = np.array([False, True])  # door 0 free, door 1 jammed
    liar = lambda i, s: int(s in (2, 3))  # claim doorways on own route
    reporter = lambda i, s: int(s in (1, 3))  # claim real jams

    board = []
    dec = fj.post_claims(5, visible, route, truth, pol, cfg, rng, board, forced=liar)
    states = {(i, s) for i, s, _, _ in dec}
    assert states == {(0, 2), (0, 1), (1, 1)}  # (agent, state) for each sighting
    assert [(c.sender, c.doorway, c.is_true) for c in board] == [
        (0, 0, False)
    ]  # one fake jam

    # no duplicate post while the claim is active; posting again after it expires
    assert (
        fj.post_claims(6, visible, route, truth, pol, cfg, rng, board, forced=liar)
        and len(board) == 1
    )
    fj.post_claims(
        5 + cfg.claim_ttl + 1, visible, route, truth, pol, cfg, rng, board, forced=liar
    )
    assert len(board) == 2

    # reporter posts the real jam; prev_visible makes each sighting count once
    board, prev = [], [set(), set(), set()]
    fj.post_claims(
        0,
        visible,
        route,
        truth,
        pol,
        cfg,
        rng,
        board,
        forced=reporter,
        prev_visible=prev,
    )
    assert sorted((c.sender, c.doorway, c.is_true) for c in board) == [
        (0, 1, True),
        (1, 1, True),
    ]
    assert (
        fj.post_claims(
            30, visible, route, truth, pol, cfg, rng, board, prev_visible=prev
        )
        == []
    )


def test_believer_never_checks():
    cfg = fj.Config(agents=3)
    doorways = [(10, 10)]
    trust = np.ones((3, 3))
    board = [fj.Claim(sender=0, doorway=0, t=0, is_true=False, audited=False)]
    avoid = np.array([[0], [20], [0]])  # robot 1 believes the fake jam
    agents = [
        (0, 0),
        (10, 11),
        (40, 40),
    ]  # robot 1 next to the doorway, robot 2 far away

    fj.reveal_checks(1, board, agents, doorways, avoid, trust, cfg)
    assert not board[0].checked  # believer never checks; far robot cannot see

    agents[2] = (10, 12)  # a non-believer comes into view
    fj.reveal_checks(2, board, agents, doorways, avoid, trust, cfg)
    assert board[0].checked
    assert (
        trust[2, 0] < 1 and trust[1, 0] == 1
    )  # only the passer-by lowers its trust in the liar

    # an audited claim is revealed immediately and lowers everyone's trust in the sender
    board, trust = (
        [fj.Claim(sender=0, doorway=0, t=0, is_true=False, audited=True)],
        np.ones((3, 3)),
    )
    fj.reveal_checks(
        0, board, [(40, 40)] * 3, doorways, np.zeros((3, 1), int), trust, cfg
    )
    assert board[0].checked and (trust[:, 0] < 1).all()


def test_detour_when_believing():
    # TODO step 6: believing "D jammed" makes the agent take the other doorway and costs the expected detour
    pass


def test_mixed_motive_window():
    # TODO step 9: with lam inside (c/B, (G-c)/(G+H)) and rule 'none', a hand-set reporter beats silence
    # in state 1 and a hand-set liar beats silence in state 2 (both incentives exist at once)
    pass


def test_claim_credit():
    from dataclasses import replace

    from tvc.tvc import TVCParams, tvc_credit

    cfg = fj.Config()
    avoiders = [
        (1, 2, 5),
        (2, 4, 1),
        (3, 1, 20),
    ]  # (agent, detour, wait): saved 3, 0 (clipped), 19 -> cap 8
    honest = fj.Claim(
        0, 0, 0, is_true=True, audited=False, avoiders=avoiders, checked=True
    )
    liar = fj.Claim(
        0, 0, 0, is_true=False, audited=True, avoiders=avoiders, checked=True
    )
    hidden_lie = fj.Claim(
        0, 0, 0, is_true=False, audited=False, avoiders=avoiders, checked=False
    )

    credit = lambda c, rule: fj.claim_credit(c, replace(cfg, rule=rule), None)
    assert credit(honest, "none") == 0.0
    assert (
        credit(liar, "popularity") == credit(honest, "popularity") == 3 * cfg.beta
    )  # pays lies too
    assert credit(honest, "accuracy") == cfg.gamma * (3 - 3 + 19)
    assert (
        credit(honest, "tvc") == 3 + 0 + cfg.tvc_params.budget + cfg.tvc_params.eta
    )  # gain (capped) + bonus
    assert credit(liar, "tvc") < 0  # bonus ~0, fined
    assert credit(hidden_lie, "tvc") == 0.0  # unchecked, unaudited
    assert credit(honest, "tvc") > credit(hidden_lie, "tvc") > credit(liar, "tvc")

    # same numbers as tvc.tvc_credit, with the claim encoded as factor 1 (true) or 2 (lie) of value 1
    for c in (honest, liar):
        saved = np.array([max(w - d, 0) for _, d, w in c.avoiders], float)
        p = TVCParams(
            **{**cfg.tvc_params.__dict__, "lock_in": 1, "p_audit": float(c.audited)}
        )
        ref = tvc_credit(
            np.random.default_rng(0),
            np.array([1.0 if c.is_true else 2.0]),
            np.array([True]),
            np.array([1.0]),
            np.zeros(len(saved), int),
            np.ones(len(saved)),
            1 - saved,
            1.0,
            p,
        )[0]
        assert np.isclose(credit(c, "tvc"), ref), (credit(c, "tvc"), ref)


def test_honesty_beats_lying_above_threshold():
    # TODO step 8 (P1 with outside temptation): with p_audit * fine > G, hand-set honest earns more
    # total reward (throughput + credit) than hand-set liar under TVC; below the threshold, the liar wins
    pass


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("fake_jams checks passed (stubs)")
