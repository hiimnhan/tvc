"""Checks for bandit.py. Run: python test_bandit.py"""
from dataclasses import replace

import numpy as np

from tvc import bandit
from tvc.tvc import TVCParams, tvc_credit


def test_credit_matches_tvc():
    # one run, 4 senders (truthful, x1.1, x2, truthful); receivers 3, 0, 1 follow senders 0, 1, 2
    fac = np.array([[1.0, 1.1, 2.0, 1.0]])
    theta, b_before, vbar = np.full((1, 4), 2.0), np.full((1, 4), 1.0), np.ones((1, 1))
    F = np.zeros((1, 4, 4), bool)
    F[0, 3, 0] = F[0, 0, 1] = F[0, 1, 2] = True
    sent = np.ones((1, 4), bool)
    for p in (0.0, 1.0):  # audits off / always on make both sides deterministic
        cfg = replace(bandit.Config(), rule="tvc", p_audit=p)
        mine = bandit.credit(cfg, np.random.default_rng(0), F, sent, fac, theta, b_before, vbar)[0]
        ref = tvc_credit(np.random.default_rng(0), (fac * theta)[0], sent[0], theta[0], np.array([0, 1, 2]),
                         theta[0, :3], b_before[0, [3, 0, 1]], 1.0, TVCParams(p_audit=p, fine=cfg.fine))
        assert np.allclose(mine, ref), (p, mine, ref)


def test_honest_naive_never_misleads():
    cfg = replace(bandit.Config(), runs=4, receiver="naive")
    pol = bandit.make_policy(cfg)
    pol.r[..., 1] = 50.0  # always report truthfully
    m = bandit.run_episode(cfg, np.random.default_rng(1), pol, np.ones((4, 10, 10)), learn=False)
    assert (m["inflation"] == 0).all() and (m["misled"] == 0).all()


def test_tvc_learns_honesty_popularity_does_not():
    infl = {}
    for rule in ("popularity", "tvc"):
        hist = bandit.train(replace(bandit.Config(), rule=rule, runs=8, episodes=150))
        infl[rule] = hist["inflation"][-30:].mean()
    assert infl["tvc"] < 0.1 < infl["popularity"], infl


if __name__ == "__main__":
    test_credit_matches_tvc()
    test_honest_naive_never_misleads()
    test_tvc_learns_honesty_popularity_does_not()
    print("bandit checks passed")
