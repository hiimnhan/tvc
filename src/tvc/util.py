"""Generic helpers: grid distances."""

import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GRID4 = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 4-neighbour moves on a 2-D grid


def chebyshev_dist(p1, p2):
    return max(abs(a - b) for a, b in zip(p1, p2))


def bfs_dist(blocked, target, steps=GRID4) -> np.ndarray:
    """Shortest-path distance (number of moves) from every cell to target, on a grid of any shape.

    blocked  bool array; True cells cannot be entered
    target   index tuple into blocked; always reachable (distance 0), even if blocked marks it
    steps    allowed moves as offset tuples (default: 4-neighbour 2-D; e.g. add diagonals or use 3-D offsets)
    Returns a float array shaped like blocked, np.inf where unreachable.
    """
    dist = np.full(blocked.shape, np.inf)
    target = tuple(target)
    dist[target] = 0
    queue = deque([target])
    while queue:
        cell = queue.popleft()
        for step in steps:
            nxt = tuple(c + s for c, s in zip(cell, step))
            if (
                all(0 <= c < n for c, n in zip(nxt, blocked.shape))
                and not blocked[nxt]
                and dist[nxt] == np.inf
            ):
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}  # two-sided 95% Student t, by degrees of freedom


def ci95(xs) -> tuple[float, float]:
    """Mean and half-width of the 95% t confidence interval (half-width 0 for a single value)."""
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    df = len(xs) - 1
    t = _T95[max(k for k in _T95 if k <= df)] if df <= 30 else 1.96
    return float(xs.mean()), float(t * xs.std(ddof=1) / np.sqrt(len(xs)))


class ProgressPool(ProcessPoolExecutor):
    """ProcessPoolExecutor whose map() prints a progress line each time a run finishes: done/total, elapsed, ETA.
    Same use as before (`with ProgressPool() as pool: list(pool.map(fn, items))`), so a long sweep shows it is alive."""

    def map(self, fn, *iterables, **kwargs):
        args = list(zip(*iterables))
        futures = [self.submit(fn, *a) for a in args]
        start, done = time.time(), [0]

        def report(_):
            done[0] += 1
            spent = time.time() - start
            left = spent / done[0] * (len(futures) - done[0])
            print(
                f"  [{fn.__name__}] {done[0]}/{len(futures)} runs done, {spent / 60:.1f} min elapsed, ~{left / 60:.1f} min left",
                flush=True,
            )

        for f in futures:
            f.add_done_callback(report)
        return (f.result() for f in futures)
