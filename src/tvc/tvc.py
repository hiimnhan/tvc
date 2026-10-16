"""Aligned verified credit (TVC): a drop-in credit rule for reliance-based rewards.

Standalone version of PAY_RULE="tvc" in sim.py (see FINDINGS.md §1b). One round, N senders.
Run: python tvc.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TVCParams:
    tau: float = 0.05  # tolerance for honest noise (use 0.05 + 2*sigma under noise)
    budget: float = 8.0  # cap on credit per follow, x mean value
    eta: float = 2.0  # scoring-bonus weight, x mean value
    p_audit: float = 0.2  # probability any sent claim is audited
    fine: float = 15.0  # fine for a failed audit, x mean value
    lock_in: int = 3  # rounds a follower is committed
    sigma: float = 0.0  # observation noise sd (sets score width)


def tvc_credit(
    rng,
    claims,
    sent,
    found,
    sender,
    follow_value,
    follow_baseline,
    mean_v,
    p=TVCParams(),
):
    """Credit for each sender this round.

    claims:          (N,) claimed values (ignored where not sent)
    sent:            (N,) bool, sender posted a claim
    found:           (N,) true value of what each sender found
    follow_sender:   (F,) int, sender index of each follow event
    follow_value:    (F,) true value of the followed arm
    follow_baseline: (F,) follower's best value before following
    mean_v:          mean arm value of the landscape (scale)
    """
    n = len(claims)
    credit = np.zeros(n)
    claims, found = np.asarray(claims, float), np.asarray(found, float)

    # 1. verified externality credit: follower's gain over lock-in, capped, forfeited if check fails
    gain = np.maximum(follow_value - follow_baseline, 0) * p.lock_in
    pay = np.minimum(gain, p.budget * mean_v)
    verified = np.abs(claims[sender] / follow_value - 1) <= p.tau
    np.add.at(credit, sender, pay * verified)

    # 2. random audits; a claim is checked if audited or followed
    audited = sent & (rng.random(n) < p.p_audit)
    checked = audited.copy()
    checked[sender] = True
    checked &= sent

    # 3. scoring bonus on checked claims, peaked at the truth
    log_err = np.zeros(n)
    log_err[sent] = np.log(claims[sent] / found[sent])
    width = max(p.sigma, 0.05)
    credit += checked * p.eta * mean_v * np.exp(-(log_err**2) / (2 * width**2))

    # 4. audit fines (ponytail: fined this round; full version would forfeit banked credit)
    caught = audited & (np.abs(np.exp(log_err) - 1) > p.tau)
    credit -= caught * p.fine * mean_v
    return credit


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    found = np.array([2.0, 2.0, 2.0, 2.0])
    claims = found * np.array(
        [1.0, 1.1, 2.0, 1.0]
    )  # truthful, x1.1, x2, truthful-unfollowed
    sent = np.ones(4, bool)
    fs = np.array([0, 1, 2])  # senders 0-2 each followed once
    fv, fb = np.full(3, 2.0), np.full(3, 1.0)  # followed arm worth 2, follower had 1

    c = tvc_credit(
        rng, claims, sent, found, fs, fv, fb, mean_v=1.0, p=TVCParams(p_audit=0.0)
    )
    assert c[0] == 3.0 + 2.0  # gain 1 x lock-in 3, plus full score bonus
    assert c[0] > c[1] > c[2]  # any lie earns less
    assert c[1] < 1.0  # x1.1 lie fails the check: no externality credit
    assert c[3] == 0.0  # unfollowed, unaudited: unchecked

    c = tvc_credit(
        rng, claims, sent, found, fs, fv, fb, mean_v=1.0, p=TVCParams(p_audit=1.0)
    )
    assert c[3] == 2.0 and c[2] < 0  # audited truth earns bonus; audited x2 lie fined

    c = tvc_credit(
        rng,
        np.array([20.0]),
        np.ones(1, bool),
        np.array([20.0]),
        np.array([0]),
        np.array([20.0]),
        np.array([1.0]),
        mean_v=1.0,
        p=TVCParams(p_audit=0.0, budget=2.0),
    )
    assert c[0] == 2.0 + 2.0  # gain 57 capped at budget 2, plus score bonus
    print("tvc checks passed")
