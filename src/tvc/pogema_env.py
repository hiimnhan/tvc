"""POGEMA environment for the fake-jam testbed: rooms joined by 1-cell doorways, jam ground truth,
moves in POGEMA action order, greedy descent on a distance map, and metric summaries.

cfg needs: rooms, room_size, agents, obs_radius, steps, on_target, jam_queue.
"""

from pathlib import Path
from typing import Any

import numpy as np
from pogema import GridConfig, pogema_v0

MOVES = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]  # POGEMA action order: stay, up, down, left, right


def make_rooms_map(cfg, rng) -> tuple[list[str], list[tuple[int, int]]]:
    """Rooms separated by 1-cell walls; one 1-cell doorway per shared wall.

    Returns (map_rows: list[str] for GridConfig(map=...), doorways: list[(row, col)] in PADDED coords).
    """
    R, s = cfg.rooms, cfg.room_size
    step = s + 1  # room interior plus one wall line
    n = R * step + 1
    wall = np.ones((n, n), bool)
    for r in range(R):
        for c in range(R):
            wall[r * step + 1 : r * step + 1 + s, c * step + 1 : c * step + 1 + s] = (
                False  # room interior
            )

    doorways = []
    for r in range(R):
        for c in range(R):
            if c + 1 < R:  # shared vertical wall with the room to the right
                d = (r * step + 1 + int(rng.integers(s)), (c + 1) * step)
                wall[d] = False
                doorways.append(d)
            if r + 1 < R:  # shared horizontal wall with the room below
                d = ((r + 1) * step, c * step + 1 + int(rng.integers(s)))
                wall[d] = False
                doorways.append(d)

    map_rows = ["".join("#" if w else "." for w in row) for row in wall]
    pad = cfg.obs_radius  # POGEMA pads the grid by obs_radius on every side
    return map_rows, [(y + pad, x + pad) for y, x in doorways]


MAPS = Path(__file__).parent / "maps"  # one text file per map: maps/<name>.txt


def load_map(name, cfg, path=None) -> tuple[list[str], list[tuple[int, int]]]:
    """Map `name` from maps/<name>.txt, one grid row per line in POGEMA's string format.

    '#' obstacle, '.' free, '!' free chokepoint ("doorway": a queue can form there; robots never start or get
    targets on it). Returns (map_rows, doorways in PADDED coords), the same as make_rooms_map.
    """
    path = Path(path) if path else MAPS / f"{name}.txt"
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    pad = cfg.obs_radius  # POGEMA pads the grid by obs_radius on every side
    return rows, [(y + pad, x + pad) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == "!"]


def make_env(cfg, map_rows, seed) -> Any:
    gc = GridConfig(
        map="\n".join(map_rows),
        num_agents=cfg.agents,
        obs_radius=cfg.obs_radius,
        on_target=cfg.on_target,
        collision_system="soft",
        max_episode_steps=cfg.steps,
        seed=seed,
    )
    env = pogema_v0(grid_config=gc)  # already wrapped with POGEMA's metrics (throughput, or ISR/CSR/ep_length)
    if cfg.on_target == "finish":
        from pogema.wrappers.metrics import SumOfCostsAndMakespanMetric

        env = SumOfCostsAndMakespanMetric(env)
    return env


def doorway_zones(doorways, walls, k) -> list[set[tuple[int, int]]]:
    """For each doorway, the set of free cells within k walking steps (4-connected, walls respected)."""
    zones = []
    for d in doorways:
        zone, frontier = {d}, [d]
        for _ in range(k):
            nxt = []
            for y, x in frontier:
                for dy, dx in MOVES[1:]:
                    q = (y + dy, x + dx)
                    if not walls[q] and q not in zone:
                        zone.add(q)
                        nxt.append(q)
            frontier = nxt
        zones.append(zone)
    return zones


def jammed(zone, agents_xy, cfg) -> bool:
    """Ground truth: at least jam_queue agents within jam_radius walking steps of the doorway (zone from doorway_zones)."""
    return sum(tuple(p) in zone for p in agents_xy) >= cfg.jam_queue


def next_action(dist, pos, rng=None) -> int:
    """Index into MOVES of the neighbour with the smallest dist (0 = stay if nothing is better).

    Ties are broken at random when rng is given, so agents do not all favour the same direction.
    Other agents are ignored here: POGEMA's collision system makes a blocked agent wait (the queue).
    """
    y, x = pos
    H, W = dist.shape
    best, choices = dist[y, x], []

    for a, (dy, dx) in enumerate(MOVES[1:], start=1):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < H and 0 <= nx < W):
            continue
        d = dist[ny, nx]
        if d == np.inf:
            continue
        if d < best:
            best, choices = d, [a]
        elif d == best and choices:
            choices.append(a)
    if not choices:
        return 0
    return int(rng.choice(choices)) if rng is not None else choices[0]


def route_doorways(dist, pos, doorways) -> list[int]:
    """Doorways on the agent's current shortest path, in order (follow dist downhill from pos)."""
    door_set = set(doorways)
    route = []
    if dist[pos] == np.inf:
        return route
    while dist[pos] > 0:
        a = next_action(dist, pos)
        pos = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
        if pos in door_set:
            route.append(doorways.index(pos))
    return route


def summarize(hist, keys, tail=50) -> dict[str, float]:
    """Mean of each metric in keys over the last `tail` episodes (hist: list of metric dicts)."""
    return {k: float(np.mean([m[k] for m in hist[-tail:]])) for k in keys}
