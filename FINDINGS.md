# Findings — Credit Where It's Due — If It's True

Status as of 2026-10-11. All results are **pilots**. There are two testbeds:
- the **bandit** collective-search pilot (§1–5b; 8–12 seeds). It ran on `sim.py`, since replaced by `src/tvc/bandit.py`.
- **POGEMA fake jams** (§5c; 10 seeds, plus scaling with 5 seeds), in `src/tvc/fake_jams.py`, `src/tvc/pogema_env.py` and `src/tvc/util.py`.

All learners are tabular. Treat every number as preliminary until it is confirmed with more seeds.

## 0. Bottom line

1. **Paying for attention breeds hype.** Under popularity credit, learners lie at every opportunity in both testbeds: about 100% inflation in the bandit, and a lie rate of 1.00 in fake jams.
2. **Hype hurts only when information drives congestion.** In the bandit, receivers who reason decode the exaggeration and welfare is unchanged (§5a–b). In POGEMA fake jams, hype is a social dilemma: it pays the hyper, but all-hype cuts deliveries by 65% (§5c i).
3. **TVC makes truth dominant whatever receivers do (P1), and learners find it.**
   - Bandit: inflation 0.3–0.4% (§4).
   - Fake jams: lie rate 0.001–0.002 for every prosocial weight λ, against 0.91–0.99 for selfish robots without credit.
   - On POGEMA's own lifelong metric, TVC gives 3.1× the throughput of no credit and 6.7× that of popularity (§5c iv).
   - The advantage over no credit grows to 4.9× on a larger map (§5c v).
4. **The fine needed is predictable.** Lying collapses between the predicted threshold $arphi_{	ext{lo}}$ and $2arphi_{	ext{lo}}$. With no fine, TVC does not stop an outside temptation (§5c iii).
5. **Limits.**
   - TVC elicits *truth*, not *usefulness*. In one-shot MAPF, its unconditional bonus causes over-reporting, and CSR falls to 0.37 against 0.87 without credit (§5c iv).
   - The bandit pilot's "half the budget" did not hold up on a matched-budget frontier (§6).
   - Fake-jam results use a calibrated per-believer temptation and naive receivers: 10 seeds on the main layout, 5 seeds across 5 layouts.
6. **Defensible claim now:** *TVC removes credit-induced and self-interested hype in a congested multi-agent setting, with a fine threshold the theory predicts.* **Not supported:** "TVC raises welfare in every task" and "TVC is budget-efficient".

## 1. Setup (bandit pilot; the fake-jams setup is in §5c)

- **Agents and landscape:** N = 10 agents, K = 50 arms with lognormal values, T = 200 rounds. Payoffs are private.
- **Messages and following:** agents post cheap-talk claims, using a ×1.0 / 1.1 / 1.5 / 2 factor or staying silent. Followers lock in for COMMIT = 3 rounds and see the true value only after following (ex-post verification).
- **Credit:** popularity β per follower, plus accuracy γ × follower gain. The popularity share is α = β/(β+γ).
- **Learners:** tabular Q-learning and softmax REINFORCE (policy gradient, PG), with B independent runs vectorised.
- **Receivers:**
  - naive: trust × claim;
  - calibrated: decode with a learned ratio, reset each episode;
  - persistent: calibrated, with memory kept across episodes.
- **Primary metric:** welfare normalised by the landscape's mean value. Raw regret misled by 3–5×, because it is dominated by landscapes with heavy-tailed values.

## 1b. The mechanism: truthful verified credit (TVC)

TVC replaces the host's "credit per follower" (popularity) with three parts.

1. **Verified externality credit.** A sender is paid for the real gain its claim gave each follower, and only if the claim checks out.
2. **Scoring bonus.** Every claim that gets checked earns a bonus that is largest when the claim is exact. This removes the sender's indifference between truth and lies.
3. **Random audits.** Claims nobody followed are also checked, at random, and lies caught this way are fined.

All amounts are scaled by the landscape's mean arm value $\bar v$, so the parameters do not depend on units. The notation follows THEORY.md §2.

### Algorithm (one round, per environment)

**Input.**
- Claims $m_i$, either a number or silence $\varnothing$. Agent $i$'s claim is about an arm of true value $\theta_i$.
- Follow events $\mathcal F = \{(j, i)\}$: receiver $j$ follows sender $i$'s claim and sees $\theta_i$.
- Each follower's best value before following, $b_j$.

**Parameters.** Tolerance $\tau$, budget cap $\bar B$, score weight $\eta$, audit probability $p$, fine $\varphi$, lock-in $L$, score width $w = \max(\sigma, 0.05)$.

**Step 1. Determine the events** for each sender $i$ with $m_i \ne \varnothing$. All later steps use these.

$$
\begin{aligned}
a_i &\sim \mathrm{Bernoulli}(p) && \text{audited (drawn first, independent of the claim)}\\
f_i &= \mathbb 1\big[\exists j : (j, i) \in \mathcal F\big] && \text{followed}\\
c_i &= \max(a_i, f_i) && \text{checked = audited or followed}\\
V_i &= \mathbb 1\Big[\Big\lvert \tfrac{m_i}{\theta_i} - 1\Big\rvert \le \tau\Big] && \text{claim passes the check}\\
e_i &= \log\frac{m_i}{\theta_i} && \text{log error of the claim}
\end{aligned}
$$

Silent senders ($m_i = \varnothing$) get $C_i = 0$ and skip the remaining steps.

**Step 2. Initialise.** $C_i \leftarrow 0$.

**Step 3. Verified externality credit**, for each follow $(j, i) \in \mathcal F$:

$$
\begin{aligned}
g_{ji} &= L\,(\theta_i - b_j)^{+} && \text{follower's realised gain over the lock-in}\\
C_i &\leftarrow C_i + V_i \cdot \min\big(g_{ji},\ \bar B\,\bar v\big) && \text{capped; forfeited if the check fails}\\
C_i &\leftarrow C_i + \gamma\,(\theta_i - b_j) && \text{host accuracy term kept in code; } 0 \text{ when } \alpha = 1
\end{aligned}
$$

**Step 4. Score bonus**, if checked ($c_i = 1$):

$$
C_i \leftarrow C_i + \eta\,\bar v\, \exp\!\Big(-\frac{e_i^2}{2w^2}\Big).
$$

**Step 5. Audit fine**, if audited and the check fails ($a_i = 1$, $V_i = 0$):

$$
C_i \leftarrow C_i - \varphi\,\bar v .
$$

Steps 3–5 only **add up** terms, so their order does not change $C_i$. What matters is that Step 1 comes first: the score bonus needs $c_i$, which needs the audit draw $a_i$.

**Output.** Agent $i$'s reward is $r_i = u_i + C_i$, its own payoff plus TVC credit.

**All in one formula:**

$$
C_i \;=\; \underbrace{\sum_{j : (j,i) \in \mathcal F} V_i \min\big(g_{ji}, \bar B\,\bar v\big)}_{\text{verified externality credit}}
\;+\; \underbrace{c_i\, \eta\,\bar v\, e^{-e_i^2/2w^2}}_{\text{score bonus}}
\;-\; \underbrace{a_i\,(1 - V_i)\,\varphi\,\bar v}_{\text{audit fine}} .
$$

### Default parameters (Gate A′)

| Symbol | Code | Value | Role |
| --- | --- | --- | --- |
| $\tau$ | `TAU` | $0.05 + 2\sigma$ | Tolerance for honest noise |
| $\bar B$ | `BUDGET` | 2, 8 or 32 | Cap on credit per follow, $\times \bar v$ |
| $\eta$ | `ETA` | 2 | Weight of the scoring bonus, $\times \bar v$ |
| $p$ | `P_AUDIT` | 0.2 | Probability that any claim is audited |
| $\varphi$ | `AUDIT_FINE` | 8 | Fine for a failed audit, $\times \bar v$ |
| $L$ | `COMMIT` | 3 | Follower lock-in |
| $w$ | — | $\max(\sigma, 0.05)$ | Score width |

Code: the pilot used `PAY_RULE` in `sim.py` (now removed; `extern`, `score` and `audit` ran one part alone). Current implementations: `tvc_credit` in `src/tvc/tvc.py`, the bandit in `src/tvc/bandit.py`, and `claim_credit` in `src/tvc/fake_jams.py`.

### Why each part is needed

| Part removed | What breaks | Observed (σ = 0) |
| --- | --- | --- |
| Verification gate (extern only) | Lies that attract followers still get paid | 88% inflation |
| Externality credit (score only) | Claims nobody followed are unchecked, so the sender is indifferent | 19% inflation |
| Scoring bonus and gate (audit only) | Fines are sparse and smooth enough that small lies pay (Kartik / T1) | 31% inflation |
| Nothing (full TVC) | — | 0.3–0.4% inflation (the budget saving was not confirmed later, §6) |

### Known simplifications

- The audit fine applies in the current round. A full version would forfeit all banked credit, which is closer to Townsend and to Ben-Porath, Dekel & Lipman.
- τ is set by hand (0.05 + 2σ), and is too loose at σ ≥ 0.1. The welfare-optimal τ is open problem T7.
- Verification assumes a follower sees the true value exactly after following. Checks that can be forged, or that are themselves noisy, are untested.

## 2. Gate 1 — basic mechanics (naive receivers)

| Finding | Evidence | Status |
| --- | --- | --- |
| Free-riding: sharing lowers exploration | 3–6× lower exploration with honest sharing (Q-learning); 0.016–0.038 vs 0.042 without communication (PG) | Robust |
| Popularity share drives inflation | Q: 13% → 86% → ~100% (α = 0 → 0.5 → ≥ 0.75); PG: 23–33% → 94–96% → ~100% | **Robust across learners** |
| Total credit raises exploration | PG: 0.016 → 0.021 → 0.037 at credit 0 → 4 → 16. Q-learning stayed flat, a learner failure exposed by the deviation test | Supported (PG) |
| Popularity share adds exploration | Only at credit 16: 0.036 vs 0.038, CIs barely separate. Explained by popularity paying *more*, not by its form | Small; budget effect |
| No strategic silence | Failure reports ≈ 100% (Q), 70–100% (PG) | H2 falsified in this model |
| Failure reports cut wasted search | Redundant exploration 0.24 with reports vs 0.75 without | Value of sharing |
| Noisy accuracy credit | Exploration unchanged; inflation at α = 0 rises from 23% to 43% | Supports H5 |

**Overturned claims.** These were raised earlier and later withdrawn:
- "Credit buys talk, not search": a Q-learning artifact.
- "Hype helps": an artifact of raw regret.
- "Forced honesty beats every fixed mix": reversed under welfare.

## 3. Gate A — verified settlement (VS) with reasoning receivers: FAIL

Setup: PG, credit 8, L = 3, 12 seeds, persistent receivers.

| Criterion | Result |
| --- | --- |
| A1 hype hurts | ❌ unwrapped 5.33 vs honest 5.40 (σ = 0); 5.34 vs 5.26 (σ = 0.1) |
| A2 VS fixes it | ❌ VS ≈ unwrapped (5.37) |
| A3 VS stops lying | ❌ inflation 36% (σ = 0), 63% (σ = 0.1) |
| A4 VS beats alternatives | ❌ accuracy pay as good or better |
| A5 VS keeps search | ✅ exploration equal |

**Why it failed:**
- VS removes the *reward* for lying but leaves senders *indifferent*: unfollowed claims are never checked.
- With naive receivers, VS *lowered* welfare (4.62 vs 5.85) by cutting follows.

## 4. Gate A′ — truthful verified credit (TVC)

TVC combines three parts:
- verified externality credit: followers' gain × lock-in, capped at a budget B̄;
- a scoring-rule bonus on every checked claim;
- random audits (p = 0.2, with a fine).

Setup: persistent receivers, PG, L = 3.

| σ | Rule | Welfare | Exploration | Inflation | Credit per exploration |
| --- | --- | --- | --- | --- | --- |
| 0 | unwrapped (α = 1) | 5.34 ± 0.30 | 0.024 | 96.5% | 3.34 |
| 0 | VS | 5.30 ± 0.46 | 0.021 | 37.7% | 3.19 |
| 0 | accuracy | 5.20 ± 0.59 | 0.026 | 79.8% | 5.86 |
| 0 | forced honest | 5.46 ± 0.36 | 0.021 | 0% | 3.60 |
| 0 | extern only | 4.52 ± 0.35 | 0.019 | 88.0% | 2.25 |
| 0 | score only | 5.16 ± 0.44 | 0.021 | 19.3% | 3.86 |
| 0 | audit only | 4.98 ± 0.32 | 0.021 | 31.1% | 3.27 |
| 0 | **TVC B̄ = 2** | 5.43 ± 0.52 | 0.019 | **0.3%** | **1.49** |
| 0 | **TVC B̄ = 8** | 5.33 ± 0.39 | 0.019 | **0.4%** | 2.73 |
| 0 | **TVC B̄ = 32** | 5.39 ± 0.31 | 0.022 | **0.4%** | 3.19 |
| 0.1 | unwrapped | 5.23 ± 0.43 | 0.023 | 95.8% | 3.62 |
| 0.1 | VS | 4.97 ± 0.41 | 0.019 | 71.2% | 3.04 |
| 0.1 | accuracy | 5.49 ± 0.48 | 0.028 | 79.4% | 6.09 |
| 0.1 | forced honest | 5.62 ± 0.54 | 0.023 | 0% | 3.45 |
| 0.1 | extern only | 4.98 ± 0.44 | 0.018 | 84.1% | 2.31 |
| 0.1 | score only | 5.81 ± 0.58 | 0.022 | 33.7% | 3.73 |
| 0.1 | audit only | 5.26 ± 0.56 | 0.022 | 38.7% | 3.12 |
| 0.1 | TVC B̄ = 2 | 4.77 ± 0.29 | 0.019 | 8.9% | 1.22 |
| 0.1 | TVC B̄ = 8 | 5.24 ± 0.50 | 0.020 | 8.2% | 2.18 |
| 0.1 | TVC B̄ = 32 | 5.08 ± 0.35 | 0.019 | 9.9% | 3.02 |

| Criterion | Result |
| --- | --- |
| A′1 welfare gain ≥ 5% | ❌ TVC ≈ every other rule |
| A′2 inflation ≤ 5% | ✅ at σ = 0 (0.3–0.4%); ❌ at σ = 0.1 (8–10%: τ = 0.25 lets the ×1.1 lie through) |
| A′3 exploration above unwrapped | ❌ slightly lower |
| A′4 welfare per credit on the frontier | ✅ in this pilot (45–55% of the credit); ❌ **not confirmed** by the matched-budget frontier in `src/tvc/bandit.py` (`figures/bandit_frontier.png`) |

**Reading.** TVC solves honesty here, but not welfare. The apparent budget saving did not hold up on a matched-budget frontier (§6). Each component alone fails, so the combination is necessary:
- extern alone pays for followers' gains but does not punish lies;
- score alone does not cover unchecked claims;
- audit alone leaves 31% inflation.

## 5. Search for a setting where hype hurts

Persistent receivers, PG, credit 8, 8 seeds. "Honest − hype" is the welfare gap.

### 5a. Costlier following

| Setting | Hype | Honest | TVC | TVC inflation |
| --- | --- | --- | --- | --- |
| Lock-in L = 10 | 4.76 ± 0.37 | 4.75 ± 0.61 | 5.09 ± 0.47 | 0.7% |
| Switching cost 1 × mean | 5.17 ± 0.73 | 4.92 ± 0.51 | 5.12 ± 0.52 | 0.5% |
| Noise σ = 0.3 | 5.36 ± 0.67 | 4.97 ± 0.29 | 5.02 ± 0.77 | 39% |
| All three | 5.00 ± 0.15 | 5.07 ± 0.37 | 5.34 ± 0.61 | 45% |

### 5b. Listener stake (a follower loses STAKE × max(0, belief − true value))

| σ | Stake | Hype | Honest | TVC | TVC inflation | Honest − hype |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 5.55 ± 0.61 | 5.29 ± 0.59 | 5.80 ± 0.33 | 0.4% | −0.26 |
| 0 | 3 | 5.70 ± 0.37 | 5.00 ± 0.69 | 4.92 ± 0.41 | 0.3% | −0.70 |
| 0.1 | 1 | 4.84 ± 0.32 | 5.76 ± 0.49 | 4.94 ± 0.60 | 10.1% | **+0.92** |
| 0.1 | 3 | 5.66 ± 0.59 | 5.12 ± 0.64 | 5.23 ± 0.40 | 9.2% | −0.54 |

**Reading:**
- Only 1 of 12 cells (σ = 0.1, stake 1) shows hype significantly hurting. That is about what chance alone would produce.
- The same setting with a *larger* stake reverses the sign, which is the opposite of what a real cost mechanism predicts.
- TVC does not recover the gap in that cell.
- Senders keep exaggerating (~95%) because the stake falls on the listener. The exaggeration is consistent, so persistent listeners decode it and over-investment losses stay small.
- With σ ≥ 0.3, TVC's tolerance τ = 0.05 + 2σ is too loose: inflation is 39–45%.

**Conclusion:** no reliable "hype hurts" setting in this testbed with reasoning receivers.

### 5c. POGEMA fake jams: hype hurts through congestion (`src/tvc/fake_jams.py`)

**Setting.** 12 robots, 3×3 rooms joined by 12 one-cell doorways, 256 steps, soft collisions, deliveries restart on arrival. A robot that sees a doorway may post "doorway $d$ is jammed". Naive believers detour for 20 steps and never check; the truth is revealed only by a passer-by who did not believe or by an audit ($p = 0.2$). Reward $r_i = (1-\lambda)\,\text{own} + \lambda\,\text{others} + C_i - c\cdot\text{posts}$, with $c = 0.05$. The claim policy is tabular REINFORCE over the state $2\cdot\text{on\_my\_route} + \text{jammed}$; movement is scripted BFS.

**(i) Hand-set policies, no learning, no credit** (`python -m tvc.fake_jams hype 50`; 50 episodes, common seeds). Robots $i < k$ post fake jams on their own route; the rest report real jams. Deliveries per robot per episode:

| hype robots $k$ | group | hype robots | honest robots | detours caused by fake jams |
| --- | --- | --- | --- | --- |
| 0 | 5.70 | – | 5.70 | 0% |
| 1 | 5.76 | 9.88 | 5.38 | 33% |
| 3 | 4.89 | 6.86 | 4.23 | 65% |
| 6 | 3.46 | 4.16 | 2.76 | 86% |
| 12 | 2.02 | 2.02 | – | 94% |

Hype pays the hyper at every $k$ (dominant strategy), honest robots lose from the first hyper, and all-hype is 65% below all-honest (Pareto-dominated): a social dilemma. At $k = 1$ the group mean is flat, because one hyper's gain (+4.2) offsets the others' loss, which is consistent with the price-of-anarchy bound (hype can help only by up to the selfish-routing gap). Figure: `figures/fake_jams_hype.png`.

**(ii) Learning, by credit rule** (`python -m tvc.fake_jams lam 150 10`; **10 seeds**, 150 episodes, last-50 mean ± 95% CI; table in `logs/fake_jams_lam.md`, figure `figures/fake_jams_lam.png`).

*Learning return.* Each claim decision gets its own effect:
$$(1-\lambda)\cdot \text{own\_gain}\cdot n_{\text{bel}} \;+\; \lambda\cdot(\pm\text{effect on others})\cdot n_{\text{bel}} \;+\; C \;-\; c,$$
and silence gets 0. The per-believer effects are calibrated from hand-set runs (`calibrate()`, 30 episodes):
- own gain $0.168$ deliveries per believer;
- harm to the others' mean $0.013$, measured in the all-liar world, because a single liar among silent robots slightly *helps* the others ($H = -0.24$ at 200 episodes, i.e. traffic spreads);
- benefit of a true report $0.088$;
- $2.3$ believers per lie.

Prediction: without credit, lying pays for $\lambda < 0.93$ (ignoring $c$).

Lie rate $P(\text{post} \mid \text{own route, free})$:

| λ | none | popularity | **TVC** |
| --- | --- | --- | --- |
| 0 | 0.992 ± 0.001 | 0.999 ± 0.000 | **0.001 ± 0.001** |
| 0.25 | 0.989 ± 0.001 | 0.998 ± 0.000 | **0.001 ± 0.000** |
| 0.5 | 0.980 ± 0.001 | 0.998 ± 0.000 | **0.001 ± 0.000** |
| 0.75 | 0.907 ± 0.002 | 0.998 ± 0.000 | **0.001 ± 0.000** |
| 1 | 0.026 ± 0.001 | 0.998 ± 0.000 | **0.001 ± 0.000** |

Throughput (deliveries per agent-step):
- none: 0.007 for λ ≤ 0.75, and 0.022 at λ = 1;
- popularity: 0.003 at every λ;
- **TVC: 0.022–0.023 at every λ**.

Spend: popularity 0.22, TVC 0.04. Useful report rate: TVC 0.98–0.99, none 0.43–0.87.

- **Selfish learners hype without credit.** They lie 91–99% of the time for λ ≤ 0.75 and stop only at λ = 1, as calibration predicts (threshold ≈ 0.93).
- **Popularity credit makes hype universal** at every λ and gives the lowest throughput.
- **TVC holds hype at 0.1% for every λ** and matches the fully cooperative no-credit benchmark (0.022). That is **about 3.2× the throughput of no credit** when agents are selfish, at about 1/5 of popularity's spend.

**(iii) Fine sweep at λ = 0** (`python -m tvc.fake_jams fine 150 10`; 10 seeds, 95% CI; `logs/fake_jams_fine.md`). Predicted threshold $\varphi_{\text{lo}} = (\text{bel}\cdot\text{own\_gain} - c)/p = 1.77$.

| fine φ | 0 | 0.44 | 0.88 | **1.77** (φ_lo) | **3.53** (2φ_lo) | 7.06 | 14.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lie rate | 0.992 ± 0.001 | 0.990 ± 0.001 | 0.986 ± 0.001 | **0.857 ± 0.025** | **0.029 ± 0.009** | 0.003 ± 0.001 | 0.002 ± 0.001 |
| throughput | 0.007 | 0.007 | 0.008 | 0.008 | **0.021** | 0.023 | 0.023 |
| useful report rate | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 |

- **The lie rate collapses between φ_lo and 2·φ_lo**, with confidence intervals that do not overlap (0.857 ± 0.025 against 0.029 ± 0.009). At φ_lo itself, lying and honesty pay almost the same, so learners switch only partly and need a margin. The theory predicts the threshold within a factor of 2.
- **With φ = 0, TVC does not stop lying.** The score bonus alone cannot beat an outside temptation. The fine is necessary (P2), and the right dominance condition is the outside-temptation form, Corollary P1-T: $p\cdot(\eta+\varphi) > \eta\bar s + (1-\lambda)G_1 - \lambda H_1 - c$.
- **The fine does not chill honest reporting:** the useful report rate stays ≥ 0.98, because honest claims are never fined when audits are exact ($q = 0$).

**(iv) POGEMA's own metrics** (`python -m tvc.fake_jams pogema 150 10`; λ = 0, 10 seeds, last-50 mean ± 95% CI; `logs/fake_jams_pogema.md`). Each task is calibrated separately.

*Lifelong MAPF* (`on_target="restart"`, POGEMA `avg_throughput` = targets per step, all agents):

| rule | lie rate | useful report rate | avg_throughput |
| --- | --- | --- | --- |
| none | 0.992 ± 0.001 | 0.43 ± 0.02 | 0.087 ± 0.002 |
| popularity | 0.999 ± 0.000 | 0.95 ± 0.01 | 0.041 ± 0.001 |
| **TVC** | **0.001 ± 0.001** | **0.99 ± 0.01** | **0.274 ± 0.009** |

On POGEMA's own metric, TVC gives **3.1×** the throughput of no credit and **6.7×** that of popularity. POGEMA's `avg_throughput` is exactly 12× our per-agent throughput.

*One-shot MAPF* (`on_target="finish"`; CSR = all agents arrive, ISR = share that arrive, SoC = sum of costs, unfinished agents counted at 256):

| rule | lie rate | useful report rate | CSR ↑ | ISR ↑ | SoC ↓ | makespan ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.903 ± 0.004 | 0.28 ± 0.04 | **0.88 ± 0.05** | **0.97 ± 0.01** | **1767 ± 94** | **147 ± 8** |
| popularity | 0.996 ± 0.001 | 0.68 ± 0.04 | 0.50 ± 0.04 | 0.85 ± 0.01 | 2754 ± 26 | 230 ± 2 |
| TVC | **0.007 ± 0.002** | 0.53 ± 0.05 | 0.35 ± 0.04 | 0.79 ± 0.01 | 2176 ± 112 | 181 ± 9 |

**Limitation: in one-shot MAPF, TVC stops lies but completes fewer tasks than no credit.**

The calibration for this task explains it:
- True reports have *negative* value: $B = -0.012$.
- Lies barely hurt: $H \approx -0.03$, $G = 0.23$.

Jams clear as agents reach their goals and leave the map, so a believer's 20-step avoidance becomes a pointless detour. Avoiding several doorways at once can leave a robot with no route at all, which is what lowers CSR.

TVC's score bonus $\eta$ pays **every checked true claim, whether or not it helped anyone**. So TVC raises reporting (0.53 vs 0.28 without credit) exactly where reporting is harmful.

**Lesson: TVC elicits truth, not usefulness.** The verified-gain term pays only for realised value, but the bonus pays for being true.

*Design options (analysed, not run):*

| bonus paid when | ignored true claim | worthless true claim | effect |
| --- | --- | --- | --- |
| always (current) | $p\eta$ | $p\eta$ to $\eta$ | no silence, but **over-reporting** when information has negative value (one-shot) |
| the claim helped (followed and $g > 0$) | 0 | 0 | no over-reporting, but **cold-start silence**: report only if $q(\eta + g) > c$, so with a low follow rate $q$ nobody reports, and with no reports nobody follows. The audit bonus exists to break exactly this |
| the information is valuable ($\hat v > 0$, a counterfactual value computed at audit, e.g. queue minus detour for robots routed through the doorway) | $p\eta$ if $\hat v > 0$, else 0 | 0 | keeps P1 wherever information has value; truth and silence tie (silence efficient) where it does not. **Needs a counterfactual-value oracle**, an extra assumption |

The value-gated bonus is the principled fix, but it is **not implemented**. It is a stated limitation and future work.

**(v) Scaling** (`python -m tvc.fake_jams scale 150 5`; λ = 0, **5 seeds**, last-50 mean ± 95% CI; each layout calibrated on its own; `logs/fake_jams_scale.md`, figure `figures/fake_jams_scale.png`). POGEMA `avg_throughput`:

| map | agents | none | popularity | **TVC** | TVC / none | TVC lie rate |
| --- | --- | --- | --- | --- | --- | --- |
| 3×3 | 8 | 0.108 ± 0.004 | 0.050 ± 0.002 | **0.216 ± 0.018** | 2.0× | 0.002 |
| 3×3 | 16 | 0.072 ± 0.004 | 0.033 ± 0.001 | **0.252 ± 0.004** | 3.5× | 0.002 |
| 3×3 | 32 | 0.045 ± 0.005 | 0.029 ± 0.002 | **0.116 ± 0.007** | 2.6× | 0.003 |
| 4×4 | 16 | 0.069 ± 0.005 | 0.021 ± 0.002 | **0.335 ± 0.005** | 4.9× | 0.001 |
| 4×4 | 32 | 0.048 ± 0.003 | 0.020 ± 0.001 | **0.220 ± 0.010** | 4.6× | 0.001 |

- **No-credit hype is robust across layouts:** the lie rate is 0.97–1.00.
- **Popularity is the worst rule at every size.**
- **TVC keeps the lie rate ≤ 0.003 everywhere.** Its throughput advantage over no credit is 2.0–4.9×, largest on the larger 4×4 map.
- **Absolute throughput falls at 32 agents for every rule:** the rooms saturate. TVC still leads by 2.6× and 4.6×.

**Caveats.**
- The private temptation is a **calibrated per-believer model**, not raw episode returns.
- Harm is taken from the all-liar world.
- Naive receivers only; seeds share one calibration per layout, so the CIs reflect learning noise rather than variety across maps.
- Main results use 10 seeds on one layout (3×3 rooms, 12 robots); scaling covers 5 layouts with 5 seeds each.
- **The fake-jam runs used TVC's default fine φ = 15** (`TVCParams.fine`), not the bandit's 8. The conclusions hold, since φ = 15 is far above φ_lo ≈ 1.8, but report φ = 15 in the paper.
- Audits are exact ($q = 0$), so the fine's upper bound is untested.
- Gain credit is still in steps, which affects spend but not lie rates.

**Bug fixed on the way.** The first λ sweep credited every decision with the whole-episode reward, and its `fake_rate` metric (the share of posts that are fake) is dominated by base rates. Both together made all rules look alike. See `.wolf/buglog.json`.

## 6. What the evidence supports

| Claim | Status |
| --- | --- |
| Credit per exploration sets exploration; the form of credit sets honesty (two levers) | Supported (PG) |
| Popularity credit causes inflation | Robust |
| Hype lowers group welfare | **Not supported** with reasoning receivers (bandit); **supported** under congestion (fake jams: all-hype −65%, popularity-trained −82%) |
| VS alone makes reports honest | **Refuted** (indifference leaves 36–71% inflation) |
| TVC makes reports honest at no welfare cost | Supported at σ = 0; partial at σ = 0.1 |
| TVC is budget-efficient | **Not supported**: the pilot showed about half the credit, but the matched-budget frontier (`bandit.py`) did not confirm it |
| Truth is dominant for every receiver rule (P1) | **Proved** (THEORY P1 and its matrix view); the outside-temptation version is confirmed in fake jams (the lie rate collapses between φ_lo and 2φ_lo) |
| Each TVC component alone is insufficient | Supported |
| TVC raises welfare or exploration | **Not supported** (bandit); **supported** in fake jams (+29% throughput over no credit, via complete honest reporting) |
| TVC raises welfare in every task | **No**: lifelong MAPF 3.1× throughput (2.0–4.9× across layouts), but one-shot MAPF CSR 0.35 vs 0.88 without credit, because the unconditional bonus causes over-reporting when information has negative value (§5c iv) |
| TVC stops an organic (non-credit) temptation to lie | **Supported** in fake jams: no-credit lie rate 0.91–0.99 (λ ≤ 0.75) → TVC 0.001; the lie rate collapses between φ_lo and 2φ_lo (§5c) |

## 7. Implications for the paper

- **Headline:** *Paying agents for attention breeds hype, which is individually rational and collectively ruinous under congestion. TVC (verified gain, a score bonus, random audits with a fine) makes truth dominant for every receiver rule. In POGEMA lifelong MAPF it drives hype from 99% to 0.2% and triples throughput, and the audit fine needed matches the predicted $\varphi_{\text{lo}}$ within a factor of 2.*
- **Honest limits for the paper:**
  - one-shot MAPF over-reporting ("truth, not usefulness"), with the value-gated bonus as future work;
  - no budget-efficiency claim;
  - the calibrated temptation model in fake jams.
- **Reframe kept from the bandit pilot.** Where receivers can decode hype, honesty matters mainly to outsiders (humans, auditors, downstream systems) who cannot.
- **Theory alignment:**
  - P1 (dominance) and its outside-temptation corollary are confirmed by the fine sweep;
  - P2 (each part needed) is supported by the bandit ablations, and by fine = 0 failing in fake jams;
  - the old T6 (hype destroys popularity credit with rational receivers) is not observed in the bandit;
  - the congestion theorems (transfer plus deadweight loss, a price-of-anarchy bound, a social dilemma) match the hand-set hype table.

## 8. Next steps

0. **Fake jams:**
   - ~~add a per-claim private term to learning~~ and ~~sweep the fine around $\varphi_{\text{lo}}$~~ (done, §5c);
   - add false-flag audits ($q > 0$) and sweep the fine to find the upper bound $\varphi_{\text{hi}}$ (chilling effect);
   - add sceptical (trust-learning) receivers;
   - run more seeds and a larger map;
   - put gain credit in delivery units (currently steps);
   - learned-agent video (TVC vs popularity after training).
   - (optional) value-gated bonus $\eta \cdot S \cdot \mathbb 1[\hat v > 0]$ to fix one-shot over-reporting; needs a counterfactual-value oracle.

1. Add an **outsider information-quality metric**: the mean |claim / true − 1| as read by a naive outsider.
2. **30-seed confirmation** of the Gate A′ table and of the σ = 0.1 / stake 1 cell, with paired tests and a Holm correction.
3. **Tune τ under noise** (T7), using held-out seeds.
4. **Learned receivers** in place of the hand-coded calibrated ones.
5. Decide whether Route A (wrapping MARL host schemes) fits the more modest claim. Then sync PROPOSAL.md, THEORY.md and ROUTE_A.md with this file.
