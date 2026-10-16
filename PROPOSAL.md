# Credit Where It's Due — If It's True

*Verified Reliance Rewards for Multi-Agent Learning* (formerly "Credit, Hype and Honest Exploration")

Research proposal · v5 · Oct 6, 2026 · @Nhan · revised after Gate 1

## Summary

**How much you pay for being followed sets how much agents search; how you pay sets how much they hype.** Paying agents when others act on their reports is the natural fix for free-riding in collective search. In simulations with learning agents (Q-learning and policy gradient), credit works on two largely separate levers:

- **Total credit** raises exploration and lowers collective regret (policy gradient: explore rate 0.016 → 0.037 and regret 5,220 → 1,310 as credit goes from 0 to 16× the mean arm value).
- **The popularity share** of credit (pay per follower vs pay for followers' realised gain) drives report inflation from ~25% to ~100%, but adds at most a small amount of exploration (~6% at high credit, none at moderate credit).

So the exploration–honesty trade-off we set out to measure is mostly *not* forced by credit design: accuracy credit buys nearly the same search as popularity credit without the hype. Two questions remain open and now carry the paper:

1. **When does hype actually cost the group?** It raised regret for Q-learners at low credit, but not for policy-gradient learners at moderate or high credit, where hype attracts more followers and spreads information.
2. **Can credit be paid without buying hype?** Verified settlement — popularity credit paid only if the claim checks out — makes truthful reporting an equilibrium of a two-agent reporting game for any popularity weight.

**Claims, in order of importance:**

1. **Verification-contingent reliance credit, and why it matters for search.** We import verification-contingent rewards (Townsend 1979; Ben-Porath, Dekel & Lipman 2014) into multi-agent reward design, where verification arises for free from peers' reliance. Unlike smooth penalties (Kartik 2009), forfeiture yields truthful reporting at bounded cost for any popularity weight; we show that with reasoning receivers it restores the exploration incentives reliance credit is meant to provide, and test it with learning agents.
2. **An empirical map of the two levers** — exploration, inflation, follower behaviour and regret as functions of total credit, popularity share and the cost of following bad tips, replicated across two learner families and checked with best-response deviation tests.
3. **An open benchmark environment** with a baseline ladder that separates what comes from the credit rule from what comes from extra information (verification, reputation).

**Setting.** Mixed-motive peer reporting during collective search, with private payoffs: agents see their own discoveries, others learn of them only through cheap-talk claims or by pulling a claimed arm themselves. It is not cooperative (unlike peer-incentivised exploration) and has no central planner controlling information (unlike incentivised exploration). Each agent's own explore/exploit choice is learned with standard RL; the paper studies the group level, where free-riding and honest reporting meet.

## Motivation

Many multi-agent systems split work into exploring and reporting: agent swarms for web or code search, automated research pipelines where components propose and test hypotheses, and human science itself. A finding matters only if others act on it, so group quality depends on what agents are rewarded for when they report.

Two failure modes are usually assumed to be linked:

- **Too little credit:** agents free-ride on others' exploration and the group searches too little.
- **Credit that tracks attention:** agents inflate weak findings, so the group chases hype.

The common intuition is that fixing the first with popularity credit worsens the second. Human science shows the pattern (citation chasing, publication bias), but reward rules cannot be varied there. In a simulated population they can, and Gate 1 shows the link is weaker than assumed: the amount of credit, not its popularity share, drives search. That makes the design problem sharper — pay enough credit to overcome free-riding, in a form that does not reward hype — and raises the question of when hype matters at all.

## Research questions and hypotheses

### Original pre-registration (Oct 5, kept unchanged as a record)

| Question | Hypothesis | Gate 1 outcome |
| --- | --- | --- |
| RQ1. How does the popularity/accuracy mix shape exploration and honesty? | H1: Raising popularity share α = β/(β+γ) raises exploration and report inflation together; collective regret is U-shaped in α. | Inflation: **supported**. Exploration: **mostly not** (small effect at high credit only). U-shape: **not found**. |
| RQ2. Do agents stop reporting failures, and what does it cost? | H2: Failure-report rate falls toward zero under popularity credit. | **Falsified** in this model: reporting is never costly to the sender. |
| RQ3. Can a mechanism beat the best fixed mix? | H3: Verified settlement lowers regret below the best fixed α\*, beyond what verification information alone explains. | Not yet tested. |
| RQ4 (secondary). How does network sparsity change these effects? | Exploratory. | Not yet tested. |

### Revised hypotheses (drive the remaining experiments)

| ID | Hypothesis | Falsified if |
| --- | --- | --- |
| H1′ | Two levers: total credit raises exploration and lowers regret; the popularity share raises inflation and changes exploration by at most a small amount. | Exploration does not rise with total credit, or the popularity share moves exploration as much as total credit does. |
| H1″ | Hype raises collective regret only when following a bad tip is costly (long lock-in, switching costs) or credit is low; otherwise extra follows offset it. | No regret effect of α at any lock-in length, or an effect that does not grow with lock-in. |
| H3 | Verified settlement cuts inflation to near zero while keeping the exploration total credit buys, and lowers regret below every fixed α and the informed baselines B1–B3 in the settings where H1″ finds hype costly. | Regret under M2 is not significantly below the best fixed α there, or B1/B3 match it. |
| H4 | Credit narrows but does not close the free-riding gap: exploration with credit stays below the no-communication level. | Some credit rule raises exploration to or above the no-communication level. |
| H5 | Noisy accuracy credit weakens honesty discipline but does not change exploration. | Exploration responds to accuracy noise, or inflation at α = 0 does not rise with noise. |
| H2′ (future work) | Strategic silence on failures requires competition between agents (e.g. rank-based credit). | — |

## Gate 1 results (Oct 6, 2026)

**Setup.** 10 agents, 50 lognormal(0, 1) arms, 200 rounds, complete graph, exploration cost c = 1 × mean arm value, noiseless observation, 1,500 training episodes, 12 seeds per cell, 95% CIs. Senders choose silent, truthful, ×1.5 or ×2 claims. Receivers follow the most trusted claim that beats their own best and are locked into a followed arm for 3 rounds; claims below a receiver's best mark that arm as a dead end. Learners: independent tabular Q-learning and tabular softmax REINFORCE with a value baseline, over the same 54 states (best-value, time-since-explore, fraction-known and recent-follower buckets). Code: originally `sim.py` (removed); now `src/tvc/bandit.py`, with the multi-agent pathfinding testbed in `src/tvc/fake_jams.py`.

| Finding | Q-learning | Policy gradient | Status |
| --- | --- | --- | --- |
| Free-riding: sharing lowers exploration | 3–6× lower with honest sharing than without | 0.016–0.038 with claims vs 0.042 without | Robust |
| Popularity share drives inflation | 13% → 86% → ~100% (α = 0 → 0.5 → ≥ 0.75) | 23–33% → 94–96% → ~100% (α = 0 → 0.5 → 1; credit 4 and 16) | **Robust across learners** |
| Total credit raises exploration | Flat (0.030) — a learner failure, see deviation test | 0.016 → 0.021 → 0.037 at credit 0 → 4 → 16 (α = 1) | **Supported (PG)** |
| Total credit lowers regret | — | 5,220 → 2,910 → 1,310 at credit 0 → 4 → 16 | Supported (PG) |
| Popularity share adds exploration | — | Credit 16: 0.036 (α = 0) vs 0.038 (α = 1), ~6%, CIs barely separate; credit 4: no reliable effect; α = 0.5 lowest in every cell | Small, high-credit only |
| Hype raises regret | Credit 1: 1,990 ± 1,000 (α = 0) vs 5,160 ± 2,000 (α = 0.75) | Credit 4: 3,350 → 2,850 → 2,450 (α = 0 → 0.5 → 1, CIs overlap); credit 16: flat (~1,220) | **Open — conditional** |
| Noisy accuracy credit (noise sd = 3 × mean value) | — | Exploration unchanged; inflation at α = 0 rises 23% → 43% (credit 16) and 33% → 46% (credit 4) | Supports H5 |
| No strategic silence | Failure-report rate ≈ 100% whenever credit > 0 | 70–100% | H2 falsified |
| Failure reports cut wasted search | Redundant exploration 0.24 with reports vs 0.75 without | — | Value of sharing |
| Forced honesty beats every fixed mix | Credit 4: 1,670 ± 490 vs ≥ 2,090 for every learned α | Not yet tested | Room for H3 (Q-learning only) |

**Deviation test (best-response check, Q-learning, α = 1).** One agent is forced to explore at 6–50% (learned ≈ 3%) against trained opponents. At credit 4 its own return never rises (−7% to +1%, within CI): low exploration is a best response. At credit 16 its return rises with exploration (+6.8 ± 6.5% at 25%, +13.7 ± 13% at 50%): the Q-learners under-explore relative to their own best response, which the policy-gradient result confirms.

**Method lessons.**
- Claims about exploration need a best-response check or a second learner family; tabular Q-learning with delayed credit under-explores.
- "No sharing without credit" was a tie-breaking artifact of greedy Q-learning; a stochastic policy keeps reporting.
- Independent learners vary strongly across seeds (single-run explore rates 0.07–0.22 in one cell); all comparisons use ≥ 12 seeds.

**What this means for the paper.** The robust result is that the popularity share buys hype. Whether that hype costs the group is conditional, and the mechanism's value in regret depends on it. Gate 2 must establish where hype is costly before M2 is tested.

## Related work and gap

| Literature | What it covers | What it leaves out | Overlap risk |
| --- | --- | --- | --- |
| Strategic experimentation (Bolton & Harris 1999; Keller, Rady & Cripps 2005) | Free-riding on others' exploration | Outcomes are public, so no reporting choice | Low |
| Private payoffs: Heidhues, Rady & Strack (2015) | Cheap talk among experimenters | Analytic, two-armed, no learning agents or credit design | Medium |
| Private payoffs: Renault, Solan & Vieille (arXiv 2512.06180) | Agents hide absence of success; over-experimentation in pure equilibria | Two-player, analytic, no credit design or learning agents | **High** — position explicitly against |
| PIMAEX, Kölle et al. (ICAART 2025, arXiv 2501.01266) | Peer-incentivised reward to boost novelty-seeking in MARL | Fully cooperative; no conflict of interest or inflation | **High** — nearest AI paper |
| Learning to incentivise other learning agents (LIO, Yang et al., NeurIPS 2020) | Agents learn to hand out rewards to shape others | No reporting channel or honesty | Medium |
| Bandit social learning with exploration episodes (arXiv 2602.05835, 2026) | Self-interested agents under-explore in a shared bandit | No report channel, no inflation | Medium |
| MARL for scholarly-publishing incentives (ISAL 2023; Frontiers in Robotics and AI 2025) | RL agents under publication incentives favour productivity over quality | Not a bandit search task, no credit-mix sweep, no inflation measure | Medium — closest in spirit |
| Truthful elicitation: peer prediction (Miller, Resnick & Zeckhauser 2005), Bayesian truth serum (Prelec 2004), proper scoring rules (Gneiting & Raftery 2007) | Payment rules that make honest reports optimal | Static one-shot elicitation, not credit for being followed during search | Medium — source of baseline B3 and of M2's design |
| Lying costs: Kartik (2009) | With smooth lying costs (intrinsic or audit penalties), no fully truthful equilibrium; inflation shrinks with cost intensity but persists | Continuous receiver actions; no reliance credit, no exploration | **High** — our Lemma 1 adapts it; T2 contrasts forfeiture (bounded, discontinuous penalty) with smooth costs |
| Costly verification: Townsend (1979); Ben-Porath, Dekel & Lipman (2014) | Rewards contingent on verified reports induce truth-telling; reward forfeited if check fails; non-negative payoffs; many agents | Principal chooses and pays for audits; static, one-period; no learning, no exploration | **High** — source of the forfeiture principle; we import it into peer-verified reliance credit |
| Incentivising exploration (Kremer, Mansour & Perry 2014; Frazier et al. 2014) | Information design to induce exploration | A central planner controls information | Low |
| Epistemic networks (Zollman 2007; Lazer & Friedman 2007) | How network structure shapes collective search | Agents report honestly | Low |
| Division of cognitive labour (Kitcher 1990; Strevens 2003) | Credit and priority in science | Little formal treatment of report honesty | Low |
| Emergent communication (Foerster et al. 2016; Lazaridou & Baroni 2020) | Agents learning to communicate | Mostly cooperative rewards | Low |
| Reward hacking / Goodhart | Proxy rewards diverge from true objectives | Single-agent focus | Low — framing for popularity credit as a proxy |

**Gap.** No work found combines decentralised peer reporting, strategic incentives to inflate, credit design, and learning agents on both sides. Based on keyword searches in Oct 2026; not proof of novelty, re-check before submission.

## Formal model

A repeated game with an action stage, a report stage and an update stage each round. The implemented version is listed first; extensions planned for robustness follow.

**Environment.** K arms with values v\_k drawn lognormal, so most arms are poor and a few are excellent. Agents i = 1…N on a graph G (complete in Gate 1). Each agent starts knowing one random arm.

**Actions.** Each round, agent i either explores (pays c, samples a random unknown arm k, observes v\_k + ε with ε ~ N(0, σ²); σ = 0 in Gate 1) or exploits (earns the value of its best known arm).

**Reports.** After exploring, i sends m\_i ∈ {silent, x = f · observation} with f ∈ {1, 1.5, 2}. Claims can be inflated but not deflated. A verifiable variant (x must equal the observation) is an ablation.

**Following.** Each receiver j follows the claim with the highest trust-weighted value w\_ji · x if it beats j's own best, is then locked into that arm for L rounds (L = 3 in Gate 1), and observes its true value. Trust is updated toward 1 if the claim was within 5% of the truth, toward 0 otherwise. Claims below j's best mark the arm as a dead end, so j does not explore it. Learned receivers are an extension.

**Payoff.** For agent i in round t:

```latex
r_i^t = u_i^t - c\,\mathbb{1}[\text{explore}] + \beta\, F_i^t + \gamma \sum_{j \in \mathcal{F}_i^t} \left(v_{k_j} - \bar{v}_j + \eta_j\right)
```

F is the number of agents who followed i's claim this round; the last term credits i for followers' gain over their previous best v̄\_j, measured with noise η ~ N(0, (s · mean value)²) (s = 0 or 3 in Gate 1). β is popularity credit, γ accuracy credit. Results are reported against total credit C = β/mean value + γ and popularity share α = β/(β+γ) at fixed C.

**Information structure.** Payoffs are private between agents: others learn of a discovery only through claims or by pulling the arm. Followers verify ex post; the credit designer observes followers' realised gains (exactly or with noise). Honest broadcast of every discovery (public payoffs, as in Bolton–Harris) is the comparison baseline.

**Group objective.** Collective regret over horizon T: the summed gap between the best arm's value and each agent's realised value.

## Theory: a two-agent reporting game

**Setup.** Sender A pays c to explore and observes y = v + ε, with v ~ F and ε ~ N(0, σ²) (σ = 0 is the exact case). A sends a claim x or stays silent. Receiver B holds a safe option worth s and follows if it believes v > s. A is paid a popularity part β if B follows plus an accuracy part γ·(v − s) if B follows. Under **verified settlement (VS)** the β part is paid only if |x − v| ≤ τ, checked after B observes v.

**Story the results support.** The obvious fix — fining liars in proportion to their error — fails (T1). The other obvious fix — paying for accuracy — fails when agents cannot be fined (T3). Verified settlement works (T2), and works as a wrapper on any reliance-based reward (T4). With receivers who reason, it does not only improve honesty: it raises exploration (T6). Under noise, the tolerance τ has an optimal setting (T7).

### Must prove (core of the paper)

| ID | Result (plain) | Formal sketch | Difficulty |
| --- | --- | --- | --- |
| **T1** | **Smooth penalties cannot stop lying.** Fining liars in proportion to how wrong they were always leaves small, profitable lies. | For any continuous fine φ with φ(0) = 0 and any β > 0, truthful reporting is not an equilibrium: for s just above v, claiming s wins β while fine and accuracy loss vanish as the gap shrinks. Inflation shrinks with fine strength (≈ √(β/λ) for a quadratic fine with γ = 0) but never reaches zero. | Easy — continuity argument near the switching threshold |
| **T2** | **Verified settlement makes honesty optimal.** If credit is forfeited when the claim fails the check, telling the truth is best however large the reward. | With σ = 0 and τ = 0, truthful reporting is an equilibrium for every β ≥ 0, γ ≥ 0, strict when γ > 0: a false claim that triggers a follow earns at most γ(v − s) ≤ 0. | Easy — pair with T1/T3 so it is not dismissed as trivial |
| **T3** | **Accuracy pay cannot stop lying if agents cannot be fined.** With non-negative payments (limited liability), lying about a bad find costs nothing, so any popularity component makes inflation pay. VS deters it using non-negative payments only. | Restrict payments ≥ 0. Under γ·max(v − s, 0) + β with any β > 0, truthful reporting is not an equilibrium; under VS with the same budget it is. | Easy–moderate — this is what separates VS from "just use accuracy pay" |
| **T4** | **Wrapper generality.** Gating any reliance-based reward by verification makes honesty a best response. | Let R_i = own payoff + S_i with S_i ≥ 0 increasing in others' reliance on i's claims. Replace S_i by S_i · 1[claim verified]. If claims affect i's payoff only through reliance, truthful reporting is a best response whenever receivers follow claims they believe. | Moderate — state conditions carefully (reliance-only channel, unforgeable verification) |

### Should prove (connects theory to the pilot findings)

| ID | Result (plain) | Formal sketch | Difficulty |
| --- | --- | --- | --- |
| **T5** | **Two levers.** How much credit is paid sets whether exploring is worth it; the form of credit sets honesty. | Under truthful reporting, A explores iff c ≤ own value + β·P(v > s) + γ·E[(v − s)⁺]: the threshold depends only on expected credit per exploration, not its split; equilibrium inflation depends on β/γ. | Moderate |
| **T6** | **With rational receivers, hype destroys popularity credit; VS restores it.** Receivers who reason ignore senders who always exaggerate, so pure popularity pay yields no follows, no credit and no extra exploration. VS restores informative claims, so credit flows and exploration rises. Explains the pilot's "hype looks harmless" result as a naive-receiver artifact. | With γ = 0 and cheap talk every equilibrium is babbling (Crawford–Sobel), credit paid = 0, so exploration equals the no-credit case. Under VS a truthful equilibrium exists with exploration incentive β·P(v > s) > 0: VS strictly raises exploration. | Moderate — babbling is standard; the comparison is new |
| **T7** | **Noise and the tolerance τ.** A strict check sometimes punishes honest agents; a loose one lets small lies through. | (a) An honest sender forfeits with probability q(τ) = P(\|ε\| > τ); (b) equilibrium inflation ≤ τ; (c) welfare-optimal τ\* increases in σ (and in β/c). | (a)–(b) easy; (c) may need numerical solution on a discretised value space |

### Nice to have / future work

| ID | Result (plain) | Formal sketch | Difficulty |
| --- | --- | --- | --- |
| T8 | **Credit alone cannot fully fix free-riding.** Credit pays only for followers' gains within a budget; the group's full benefit is larger. | Whenever the credit budget is below the social value of a discovery, equilibrium exploration is below the planner's optimum. Explains H4. | Easy — standard argument |
| T9 | **Common-value arms bring back strategic silence.** When failures inform about a shared option, hiding them can pay (link to Renault, Solan & Vieille). | Requires a shared risky arm with stochastic outcomes; beyond the current model. | Hard — follow-up theory paper |

**Priority and effort.** Must (T1–T4): 1–2 weeks of short proofs and careful statements. Should (T5, T6, T7 a–b): 2–3 weeks. Optional (T7 c numerical, T8): about 1 week. T9: separate paper. If any proof stalls, solve the two-agent game numerically on a discretised value space and report the result as a numerical proposition.

**Key messages.** T1 + T3: the two obvious fixes fail — proportional fines always leave profitable small lies, and accuracy pay cannot deter lies without fines. T2 + T4: credit that is *forfeited* on a failed check works, for any reliance-based reward. T6: with sensible receivers this is not only about honesty — it is what makes reliance credit buy exploration at all.

## Mechanisms

| Mechanism | How it works | Failure it targets | Free parameters |
| --- | --- | --- | --- |
| **M2. Verified settlement** (headline) | Popularity credit is held until followers observe the true value and paid only if the claim is within τ of it; otherwise forfeited | Inflation | Tolerance τ (scaled to noise σ) |
| M1. Reputation-weighted credit | Popularity credit scales with the sender's rolling accuracy | Inflation | Window length, newcomer trust |
| M4. Newcomer trust floor | Trust floored at w\_min for senders with fewer than T\_min reports | Entrenchment caused by M1 | w\_min, T\_min |
| M3. Failure credit | Reward δ for reporting an arm below the sender's best | Silence on dead ends | δ — **deprioritised**: Gate 1 found no silence (H2 falsified) |

**Baseline ladder** (separates credit design from information):

| Baseline | Credit | Information |
| --- | --- | --- |
| B0. Fixed α | Best fixed mix α\* at the same total credit | None beyond the base model |
| B1. Fixed α + verification | Fixed mix | Same post-hoc verification signal as M2 |
| B2. Reputation only | Fixed mix | Receivers use reputation (no change to credit) |
| B3. Proper-scoring payment | Peer-prediction-style scoring against realised follower value | Same verification signal |
| Honest reporters | Forced truthful reports at the same total credit | Upper bound on the value of honesty |

**What counts as winning.** M2 wins if, in settings where hype is costly (H1″), it lowers collective regret below B0, B1 and B3 at the same total credit, keeps exploration at the level total credit buys, and holds on at least three of four network types. A win only on the default setting, or a null result because hype is cheap, is reported as such.

**Tuning protocol.** Each mechanism's one or two parameters are tuned by grid search on 10 held-out seeds at the default setting, frozen, then evaluated on fresh seeds. Baselines B1 and B3 get the same tuning budget.

## Implementation

A single NumPy module (originally `sim.py`, now `src/tvc/bandit.py`) runs B independent runs side by side (seeds × conditions), each with its own world and learner tables, so learning is exactly sequential per run. A full α × seed sweep at one setting takes 2–7 minutes on a laptop.

| Parameter | Gate 1 value | Planned sweep |
| --- | --- | --- |
| Agents N | 10 | 10, 20, 50 |
| Arms K | 50 | 50, 100, 500 |
| Arm values | Lognormal(0, 1) | Lognormal σ 0.5–2 |
| Exploration cost c | 1 × mean arm value (0.2 and 3 in step 2) | 0.2–3 × mean |
| Observation noise σ | 0 | 0–0.5 × SD |
| Accuracy-credit noise s | 0, 3 | 0–5 × mean |
| Follower lock-in L | 3 | **1, 3, 10** (Gate 2) |
| Total credit C | 0, 1, 4, 16 | 0–16 |
| Popularity share α | 0, 0.25, 0.5, 0.75, 1 | 0 to 1 in steps of 0.25 |
| Horizon T | 200 | 200–1000 |
| Network | Complete | Complete, ring, small-world, random |
| Seeds per cell | 12 | 12–30 |

**Learners.** Independent tabular Q-learning (ε-greedy) and tabular softmax REINFORCE with a value baseline, both over 54 states. Policy gradient is the primary learner from Gate 2 on, because Q-learning under-explores at high credit; Q-learning is kept as a robustness check. A deviation test (one agent forced to a different explore rate against trained opponents) checks best responses for any exploration claim.

## Experimental plan

1. **Gate 2 — when is hype costly? (H1″).** Policy gradient, α ∈ {0, 0.5, 1} × total credit {1, 4, 16} × lock-in L ∈ {1, 3, 10}, plus forced-honest reporters at each cell. Locate settings where regret rises with α and honest reporting beats every fixed α.
2. **Mechanism test (H3).** Implement M2 (and M1 as a comparison) and evaluate against the baseline ladder in the costly-hype settings from experiment 1, with the tuning protocol above. Both learners.
3. **Two-lever map (H1′, H4, H5).** Full sweep over total credit and α, with accuracy noise, at the default setting; deviation tests at key cells.
4. **Robustness.** Repeat the main comparisons across network type, landscape ruggedness, exploration cost, observation noise and N.
5. **Theory check.** Compare two-agent equilibrium predictions (Props. 4–7) with learned two-agent play.
6. **Extensions (appendix or future work).** Learned receivers; deflatable claims (removing the credibility of low claims); competition between agents for strategic silence (H2′); non-stationary landscape.

**Ablations.** Cheap talk vs verifiable reports; Q-learning vs policy gradient; fixed vs learned receiver trust.

## Metrics and analysis

| Metric | Definition | Role |
| --- | --- | --- |
| Collective regret | Sum over agents and rounds of (best arm value − realised value); absolute and normalised by no communication | Primary outcome |
| Exploration rate | Share of agent-rounds spent exploring | Lever 1 (total credit) |
| Report inflation | Share of sent reports that are inflated | Lever 2 (popularity share) |
| Follows per episode | Number of follow decisions | Channel through which hype can help or hurt |
| Redundant exploration | Share of explorations hitting arms someone else already found | Value of sharing |
| Failure-report rate | Share of below-best discoveries that are reported | H2′ |
| Deviation gain | Change in one agent's return when forced to explore more | Best-response check |
| Credit concentration | Gini of total credit across agents | Entrenchment (M1) |

**Statistics.** Means with 95% CIs over seeds (≥ 12). Mechanism comparisons use paired tests across matched seeds with Holm correction. Effects of α and total credit are tested with linear trends across levels plus pairwise contrasts at the extremes; H1″ is tested as an α × lock-in interaction. Hypotheses and analysis plan for Gate 2 are fixed before it runs.

## Expected contributions

1. **Verification-contingent reliance credit for multi-agent reward design** (headline): importing the forfeiture principle of the costly-verification literature into a setting where verification is free and triggered by peers' reliance; a new result that, with reasoning receivers, forfeiture restores the exploration reliance credit is meant to buy; noisy-check tolerance; and evidence with learning agents. The truth-telling property itself builds on Townsend (1979), Ben-Porath, Dekel & Lipman (2014) and Kartik (2009) and is not claimed as new.
2. **The two-lever result:** total credit buys search; the popularity share buys hype and little else. Replicated across two learner families with best-response checks, and with a characterisation of when hype costs the group.
3. **A benchmark environment**, open source, with a baseline ladder that separates credit design from information.
4. **A methodological caution for MARL incentive studies:** exploration claims from tabular Q-learning can reverse under policy gradient; best-response deviation tests catch this.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Hype is cheap in every realistic setting, so M2 cannot lower regret | **Medium–High** | Gate 2 decides this first. If hype is never costly, the paper becomes the two-lever result plus the honesty guarantee, with honesty valued for downstream users rather than regret |
| The mechanism win comes from extra information, not credit design | Medium | Baseline ladder B1–B3; report which component does the work |
| Results depend on the learner | Medium (already observed) | Policy gradient primary, Q-learning secondary, deviation tests for every exploration claim |
| Results depend on modelling choices (lock-in, no deflation, fixed receivers) | Medium | Sweep lock-in; add deflatable claims and learned receivers as extensions |
| Learning varies strongly across seeds | High (observed) | ≥ 12 seeds, CIs on every number |
| The noisy-case result (τ\*) is hard to prove | Medium | Props. 4–5 carry the headline; solve Props. 6–7 numerically if needed |
| Reviewers call Prop. 5 trivial | Medium | Lead with Prop. 4 (the natural fix provably fails) and Prop. 7 (when honesty matters) |
| A recent paper covers the same idea | Medium | Re-run the literature check before submission |

## Timeline and milestones

About six months part-time to a submittable paper.

| Month | Work | Gate |
| --- | --- | --- |
| 1 ✅ | Environment, vectorised simulator, Q-learning and policy-gradient learners, Gate 1 sweeps, deviation test | **Gate 1 (done):** inflation emerges; exploration is driven by total credit, not popularity share; original H1 trade-off mostly not supported |
| 2 | Gate 2 sweep (lock-in × credit × α), two-agent theory (Props. 4, 5, 7) | **Gate 2:** is there a setting where hype costs regret? If not, switch to the fallback framing in Risks |
| 3 | M2 and M1 implementation, tuning on held-out seeds | |
| 4 | Mechanism evaluation on fresh seeds; two-lever map | |
| 5 | Robustness, ablations, extensions | |
| 6 | Writing, figures, literature re-check, code release | |

Optional: a workshop paper after month 4.

## Target venues and writing plan

| Stage | Venue | Content |
| --- | --- | --- |
| 1 (optional) | Multi-agent or cooperative-AI workshop at NeurIPS, ICML or ICLR | Two-lever result and mechanism, 4 pages |
| 2 | NeurIPS / ICML / ICLR main track, or AAMAS | Full paper |
| Alternative | AAAI, IJCAI; EC for an economics-leaning framing | Same paper, reframed |

**Paper outline (8–9 pages).** Introduction stating the two levers in the first paragraph (1) · Related work (0.75) · Model and environment (1.25) · Theory: two-agent game, smooth penalties fail, verified settlement, when hype is costly (1.25) · The two levers (1.25) · When hype costs the group (0.75) · Mechanism and baseline ladder (1.5) · Robustness and learner dependence (0.5) · Discussion and limitations (0.75).

**Lead figure.** Two panels: (a) exploration rate against total credit, one line per α — lines rise together and nearly overlap; (b) inflation against α, one line per total credit — lines rise together. Together they show the two levers are separate. A third panel adds regret against α for short and long lock-in, with M2 as a point below the costly-hype curve.

## Key references

Compiled from memory and Oct 2026 search; **verify authors, volume and pages before citing.** Entries marked (to verify) need their full details looked up.

- Bolton, P. & Harris, C. (1999). Strategic experimentation. *Econometrica*, 67(2).
- Crawford, V. & Sobel, J. (1982). Strategic information transmission. *Econometrica*, 50(6).
- Foerster, J., Assael, Y., de Freitas, N. & Whiteson, S. (2016). Learning to communicate with deep multi-agent reinforcement learning. *NeurIPS*.
- Frazier, P., Kempe, D., Kleinberg, J. & Kleinberg, R. (2014). Incentivizing exploration. *ACM EC*.
- Gneiting, T. & Raftery, A. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477). (to verify)
- Heidhues, P., Rady, S. & Strack, P. (2015). Strategic experimentation with private payoffs. *Journal of Economic Theory*, 159.
- Keller, G., Rady, S. & Cripps, M. (2005). Strategic experimentation with exponential bandits. *Econometrica*, 73(1).
- Kitcher, P. (1990). The division of cognitive labor. *Journal of Philosophy*, 87(1).
- Kölle, M., Tochtermann, J., Schönberger, J., Stenzel, G., Altmann, P. & Linnhoff-Popien, C. (2025). PIMAEX: Multi-agent exploration through peer incentivization. ICAART 2025. arXiv:2501.01266.
- Kremer, I., Mansour, Y. & Perry, M. (2014). Implementing the "wisdom of the crowd". *Journal of Political Economy*, 122(5).
- Lazaridou, A. & Baroni, M. (2020). Emergent multi-agent communication in the deep learning era. arXiv:2006.02419.
- Lazer, D. & Friedman, A. (2007). The network structure of exploration and exploitation. *Administrative Science Quarterly*, 52(4).
- Miller, N., Resnick, P. & Zeckhauser, R. (2005). Eliciting informative feedback: the peer-prediction method. *Management Science*, 51(9). (to verify)
- Prelec, D. (2004). A Bayesian truth serum for subjective data. *Science*, 306. (to verify)
- Renault, J., Solan, E. & Vieille, N. (2026, forthcoming). Strategic experimentation with privately observed payoffs. *Theoretical Economics*. arXiv:2512.06180.
- Strevens, M. (2003). The role of the priority rule in science. *Journal of Philosophy*, 100(2).
- Weisberg, M. & Muldoon, R. (2009). Epistemic landscapes and the division of cognitive labor. *Philosophy of Science*, 76(2).
- Yang, J., Li, A., Farajtabar, M., Sunehag, P., Hughes, E. & Zha, H. (2020). Learning to incentivize other learning agents. *NeurIPS*. (to verify)
- Bandit social learning with exploration episodes (2026). arXiv:2602.05835. (to verify authors and venue)
- Examining the role of incentives in scholarly publishing with multi-agent reinforcement learning (ISAL 2023); and the 2025 *Frontiers in Robotics and AI* paper on the same topic. (to verify authors)
- Zollman, K. (2007). The communication structure of epistemic communities. *Philosophy of Science*, 74(5).



