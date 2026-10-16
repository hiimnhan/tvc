# Theory plan: Credit Where It's Due — If It's True

Working notes · rewritten Oct 8, 2026 around truthful verified credit (TVC) · companion to `FINDINGS.md` (results), `PROPOSAL.md` and the code: `src/tvc/tvc.py` (TVC), `src/tvc/bandit.py` (bandit testbed), `src/tvc/fake_jams.py` with `src/tvc/pogema_env.py` (POGEMA fake jams).

This file covers what to prove, how to approach each proof, and what to read. References are from memory, so **verify every citation before use**. Maths uses LaTeX (`$…$` inline, `$$…$$` display).

## 1. What changed and why

The earlier plan (T1–T9) was built around verified settlement (VS) and the conjecture that hype destroys popularity credit when receivers reason (T6). The pilots changed the picture (FINDINGS.md §3–5):

| Pilot finding | Consequence for theory |
| --- | --- |
| VS alone left 36–71% inflation | VS makes senders only **weakly** truthful: unfollowed claims are never checked, so lying and truth pay the same. It becomes a background lemma (L2), not a main result. |
| With reasoning receivers, hype ≈ honest on welfare (11 of 12 cells) | T6 is not observed. It is dropped and replaced by conjecture C1 (senders who inflate by different amounts). |
| TVC: 0.3–0.4% inflation at equal welfare; each component alone fails. The pilot's "half the credit" was not confirmed on a matched-budget frontier | The theory now centres on TVC: **why it is truthful (P1), why every part is needed (P2), why learners find the truth (P3), and why its pay is bounded by the value created (P4)**. P4 is a safety bound, not an efficiency claim. |

## 2. Formal setting

We start with the smallest version, **one sender, one receiver and one round** (§2.0), where every idea is visible. Sections 2.1–2.7 then generalise it to many agents, many rounds and noise.

### Notation

| Symbol | Meaning | Code (`tvc.py` / `fake_jams.py`) |
| --- | --- | --- |
| **Environment** | | |
| $n,\ N = \{1,\dots,n\}$ | number of agents, set of agents | `cfg.agents` |
| $K,\ v_k$ | number of arms, value of arm $k$ | bandit only |
| $\bar v$ | landscape scale, mean arm value; unit for $\bar B, \eta, \varphi$ | `mean_v` (fake jams: 1 step) |
| $t,\ T$ | round, horizon | `t`, `cfg.steps` |
| $\mathcal K_i^t,\ b_i^t$ | arms known to $i$, value of $i$'s current best | — |
| $c$ | exploration cost (bandit) / report cost (fake jams) | `cfg.report_cost` |
| $u_i^t$ | $i$'s own payoff | `throughput[i]` |
| $y_i,\ \varepsilon_i,\ \sigma$ | noisy observation $y_i = v_k e^{\varepsilon_i}$, noise, noise level | `sigma` |
| $\theta_i^t$ | true value of what $i$ claims about | `claim.is_true` |
| **Claims and reliance** | | |
| $m_i^t,\ \varnothing$ | $i$'s claim, silence | posts on `board` |
| $m^t$ | all claims in round $t$ | `board` |
| $\kappa_i^t = m_i^t / y_i^t$ | inflation factor ($1$ = truthful) | — |
| $d_j^t$ | whom receiver $j$ follows ($0$ = nobody) | `avoid`, `claim.avoiders` |
| $\rho_j$ | receiver rule: $d_j^t = \rho_j(h_j^t, m^t)$ | `cfg.receiver` |
| $h_j^t$ | $j$'s history before round $t$ | `trust` |
| $L$ | lock-in: rounds a follower stays committed | `lock_in` / `avoid_steps` |
| $g_{ji}^t = L(\theta_i^t - b_j^t)^+$ | follower $j$'s gain from $i$'s claim | `wait - detour` |
| **Host schemes** | | |
| $H_i^t$ | host's reliance reward | `rule` |
| $\beta$ | popularity weight (pay per follower) | `cfg.beta` |
| $\gamma$ | accuracy weight (pay per unit of gain) | `cfg.gamma` |
| $\alpha = \beta/(\beta+\gamma)$ | popularity share | — |
| **TVC** | | |
| $\tau$ | tolerance: claims within $\tau$ count as verified | `tau` |
| $\bar B$ | cap on credit per follower (units of $\bar v$) | `budget` |
| $\eta$ | score-bonus weight | `eta` |
| $p$ | audit probability | `p_audit` |
| $\varphi$ | fine per caught lie (units of $\bar v$) | `fine` |
| $w$ | score width, $\max(\sigma, 0.05)$ | `sigma` floor |
| $a_i^t \sim \mathrm{Bernoulli}(p)$ | audited | `claim.audited` |
| $f_i^t$ | followed by someone | `len(claim.avoiders) > 0` |
| $c_i^t$ | checked (audited or followed / passer-by) | `claim.checked` |
| $V_i^t$ | verified: $\lvert m/\theta - 1\rvert \le \tau$ | `claim.is_true` |
| $S(m,\theta)$ | score $\exp(-(\log m - \log\theta)^2 / 2w^2)$ | `score` |
| $\bar s$ | best score any lie can earn | — |
| $C_i^t$ | TVC credit | `claim_credit`, `tvc_credit` |
| $r_i^t$ | total reward $u_i^t + C_i^t$ | `reward` |
| **Fake jams (mixed motive)** | | |
| $\lambda$ | prosocial weight: $r_i = (1-\lambda)\,\text{own} + \lambda\,\text{others} + C_i - c\cdot\text{posts}$ | `cfg.prosocial` |
| $B$ ($B_1$) | others' gain from true reports (per report) | `measure_B_G_H` |
| $G$ ($G_1$) | liar's own gain from lying (per lie) | `measure_B_G_H` |
| $H$ ($H_1$) | others' harm from lying (per lie) | `measure_B_G_H` |
| $n_{\text{lie}}$ | lies per episode, $G_1 = G / n_{\text{lie}}$ | not yet logged |
| $q$ | false-flag rate of audits | not yet in code |
| $\varphi_{\text{lo}},\ \varphi_{\text{hi}}$ | fine bounds: $\frac{(1-\lambda)G_1 - \lambda H_1 - c}{p} < \varphi < \frac{\lambda B_1 + P_{\text{chk}}(\text{gain}+\eta) - c}{p\,q}$ | — |
| **Objectives and learning** | | |
| $W$ | welfare $\frac{1}{\bar v}\sum_{i,t} u_i^t$ (transfers excluded) | `throughput` |
| $\mathcal B$ | total credit budget | `spend` |
| $\epsilon$ | allowed inflation rate | `fake_rate` |
| $\Delta$ | dominance margin (truth's advantage) | — |
| $\mathrm{Reg}_i(T)$ | learner's regret | — |

### 2.0 Simple setup: one sender, one receiver, one round

**Players.**
- A **sender** $S$ has found an arm of true value $\theta > 0$ and knows it exactly.
- A **receiver** $R$ has a safe option worth $b > 0$.

**Timing.**
1. $S$ sends a claim $m \in \{\varnothing\} \cup \mathbb R_{>0}$, either silence or a number. The claim is **truthful** if $m = \theta$.
2. $R$ decides whether to follow, $d = \rho(m) \in \{0, 1\}$, using **any** rule $\rho$.

   **What $\rho$ is.** The receiver's *decision rule*: a function that maps the claim it hears to "follow" ($1$) or "stay" ($0$). $R$ can only act on the claim, because it does not know $\theta$ until it follows. Examples:

   | Rule | $\rho(m)$ | Behaviour |
   | --- | --- | --- |
   | Naive | $\mathbb 1[m > b]$ | Takes the claim at face value; follows if it beats the safe option |
   | Trusting (testbed "trust") | $\mathbb 1[\text{trust}\cdot m > b]$ | Discounts every claim by a fixed factor |
   | Calibrated (testbed "calibrated" / "persistent") | $\mathbb 1[m/\hat\kappa > b]$ | Divides by the inflation $\hat\kappa$ it has learned to expect from $S$ |
   | **Bayesian** | $\mathbb 1\big[\mathbb E[\theta \mid m] > b\big]$ | Updates a prior over $\theta$ using the claim and a model of how $S$ reports; follows if the expected value beats the safe option (details below) |
   | Sceptical | $0$ | Never follows |
   | Credulous | $\mathbb 1[m \ne \varnothing]$ | Follows any claim |
   | Learned | output of $R$'s own RL policy | Changes over training |
   | Adversarial | chosen to make lying look best | Worst case for honesty, e.g. follow only inflated claims |

   **The Bayesian rule in detail.** $R$ has a prior $\theta \sim F$ with density $f$, and a belief $\sigma_S(m \mid \theta)$ about how $S$ turns the truth into a claim (its reporting strategy). On hearing $m$, $R$ forms the posterior by Bayes' rule and follows if the expected value beats the safe option:

   $$
   \pi(\theta \mid m) = \frac{f(\theta)\,\sigma_S(m \mid \theta)}{\int f(\theta')\,\sigma_S(m \mid \theta')\,d\theta'},
   \qquad
   \rho_{\text{Bayes}}(m) = \mathbb 1\Big[\int \theta\, \pi(\theta \mid m)\, d\theta > b\Big].
   $$

   What it does under each credit rule:

   | Sender's strategy | Posterior | Bayesian receiver |
   | --- | --- | --- |
   | Truthful, $m = \theta$ (TVC) | Point mass at $m$ | Follows iff $m > b$, the same as the naive rule. Claims carry full information |
   | Constant inflation, $m = k\theta$ | Point mass at $m/k$ | Follows iff $m/k > b$, the same as the calibrated rule. Decodes the hype (Kartik's "language inflation") |
   | Every type sends the same claim (popularity credit with cheap talk) | Equal to the prior | Follows iff $\mathbb E_F[\theta] > b$, ignoring the claim. **Babbling** (Crawford–Sobel): the claim is useless |
   | Inflation varies across types or senders | Spread out | Partially informative; welfare loss (Frankel & Kartik; conjecture C1) |

   **How it connects to P1.**
   - A Bayesian receiver is just one row of the table above, so P1 covers it: truth is still best for $S$.
   - Because truth is dominant, a Bayesian $R$ who expects truth is *correct* to do so. "$S$ truthful, $R$ Bayesian with $\sigma_S(m \mid \theta) = \mathbb 1[m = \theta]$" is therefore a **perfect Bayesian equilibrium**, and in it the Bayesian rule reduces to the naive rule.
   - Under popularity credit, a Bayesian $R$ cannot use the claims at all (babbling), or has to decode them. TVC makes the claim mean exactly what it says, so even an outsider without a model of $S$ can read it (C2).

   **Why "any" rule.** We do **not** assume $R$ is rational or Bayesian, or has a particular rule. The result (truth is best for $S$) must hold for every row of this table at once. This matters because:
   - in MARL, receivers are themselves learning, so their rule keeps changing;
   - we do not want the guarantee to depend on a model of receivers that might be wrong;
   - it explains why TVC worked equally well with naive, calibrated and persistent receivers in the pilots.

   In game-theory terms, this makes truth a **dominant strategy**, not just an equilibrium strategy (which would assume a specific $\rho$).
3. If $R$ follows, $R$ switches to $S$'s arm and sees $\theta$. This is the check. $R$'s gain is
   $$g = (\theta - b)^{+}.$$
4. Independently, the designer audits $S$'s claim with probability $p$: $a \sim \mathrm{Bernoulli}(p)$.

**Host reward (popularity credit).** $S$ is paid $H(m) = \beta\, d$, so $S$ earns $\beta$ whenever $R$ follows, true or not.
- *Problem:* if $\theta < b$, a truthful claim gets no follow and earns $0$. An inflated claim $m > b$ may get followed and earn $\beta > 0$. **Lying pays.**

**TVC reward.** Let

$$
V = \mathbb 1[m = \theta] \quad \text{(claim is true)}, \qquad
c = \mathbb 1[m \ne \varnothing]\cdot \max(a, d) \quad \text{(claim is checked)} .
$$

Then

$$
C(m) \;=\; \underbrace{d\, V \min(g, \bar B)}_{\text{paid for real gain, only if true}}
\;+\; \underbrace{c\, \eta\, S(m, \theta)}_{\text{bonus if checked}}
\;-\; \underbrace{a\,(1 - V)\,\mathbb 1[m \ne \varnothing]\, \varphi}_{\text{fine if audited and false}}
$$

with score $S(\theta, \theta) = 1$ and $S(m, \theta) \le \bar s < 1$ for every lie $m \ne \theta$.

**Payoffs case by case** (amounts only; silence always earns $0$):

| Case | Truth, $m = \theta$ | Lie, $m \ne \theta$ |
| --- | --- | --- |
| Followed, audited | $\min(g, \bar B) + \eta$ | $\eta\,S(m,\theta) - \varphi$ |
| Followed, not audited | $\min(g, \bar B) + \eta$ | $\eta\,S(m,\theta)$ |
| Not followed, audited | $\eta$ | $\eta\,S(m,\theta) - \varphi$ |
| Not followed, not audited | $0$ | $0$ |

**What the table shows.**
- In every row, truth earns at least as much as the lie, and strictly more except in the last row.
- The worst case for truth is "not followed": it earns $\eta$ with probability $p$, so $\mathbb E[C(\theta)] \ge p\eta$.
- The best case for a lie is "followed": it earns at most $\eta\bar s - p\varphi$ in expectation.
- So **truth beats every lie, whatever the receiver does**, when
  $$p\,\eta \;>\; \eta\,\bar s - p\,\varphi \iff p \cdot (\eta + \varphi) > \eta\,\bar s .$$
  This is P1 in its simplest form.

**Where each part matters** (P2 in miniature):
- Without the gate ($V$), a followed lie still collects $\min(g, \bar B)$.
- Without audits ($p = 0$), the "not followed" row is $0$ for both, so $S$ does not care whether it lies (verification alone, L2).
- Without the bonus ($\eta = 0$), truth still beats lies by $p\varphi$, since the fine alone does that. But a truthful claim that is not followed earns $0$, the same as **silence**, so $S$ has no reason to report a bad find. The bonus is what makes reporting failures strictly better than staying silent.
  - *Note:* this means P2's "every part is necessary" needs restating. The fine handles **lies**; the bonus handles **silence**. A run with `tvc` minus the score has not been done yet.

**From simple to full.** The general setting below changes five things:

| Simple setup | Full setting (§2.1–2.7) |
| --- | --- |
| 1 sender, 1 receiver | $n$ agents, each both sender and receiver |
| 1 round | $T$ rounds; following locks the receiver in for $L$ rounds, so $g = L(\theta - b)^+$ |
| Arm already found | Agents choose to explore (paying cost $c$) or exploit |
| Exact values, check requires $m = \theta$ | Noisy observation $y = \theta e^{\varepsilon}$; check allows tolerance $\tau$ |
| Payoffs in absolute units | Amounts scaled by the mean arm value $\bar v$ |

### 2.1 Environment: collective search with private payoffs

- **Agents and options.** There are $n$ agents, $N = \{1, \dots, n\}$, and $K$ arms. Each arm has a value $v_k > 0$, drawn i.i.d. from a distribution $F$ with finite mean (lognormal in the testbed). Values are fixed within an episode. The landscape scale is
  $$\bar v = \frac{1}{K}\sum_{k=1}^{K} v_k .$$
- **Rounds.** $t = 1, \dots, T$. Agent $i$ holds a **known set** $\mathcal K_i^t \subseteq [K]$ and a **current best**
  $$b_i^t = \max_{k \in \mathcal K_i^t} v_k .$$
- **Actions.** In each round, each agent chooses, through a learned policy $\pi_i$:
  - *explore:* pull an unknown arm, paying cost $c \ge 0$, and add it to $\mathcal K_i$;
  - *exploit:* pull its current best;
  - while committed to a follow (below), its action is fixed.
- **Private payoffs.** Agent $i$ observes the payoff of its own pulls only. Its **own payoff** $u_i^t$ is the value of the arm it pulls, minus $c$ when exploring.
- **Noisy observation.** When $i$ discovers arm $k$, it observes
  $$y_i = v_k\, e^{\varepsilon_i}, \qquad \varepsilon_i \sim \mathcal N(0, \sigma^2) \text{ i.i.d.}$$
  $\sigma = 0$ is the exact case. Write $\theta_i^t$ for the true value of the arm $i$ reports on.

### 2.2 Claim channel and reliance

- **Claims.** In each round, each agent may send a public claim $m_i^t \in \mathcal M = \{\varnothing\} \cup \mathbb R_{>0}$. $\varnothing$ means silence. A claim is **truthful** if $m_i^t = y_i^t$. The **inflation factor** is $\kappa_i^t = m_i^t / y_i^t$.
- **Reliance.** Each receiver $j$ chooses at most one claim to follow: $d_j^t \in \{0\} \cup N$, where $0$ means none.
  - $d_j^t = \rho_j(h_j^t, m^t)$ for a **receiver rule** $\rho_j$ that may depend arbitrarily on $j$'s history $h_j^t$ and on all current claims $m^t = (m_1^t, \dots, m_n^t)$. The rule may be naive, calibrated, learned or adversarial.
  - Following $i$ commits $j$ to $i$'s arm for $L$ rounds.
- **Verification by reliance.** A follower observes $\theta_i$ exactly upon committing. The designer observes $\theta_i$ on every checked claim.
- **Follower gain.** This is $j$'s realised improvement over its own best during the lock-in, measured against the counterfactual of staying put:
  $$g_{ji}^t = L\,\big(\theta_i^t - b_j^t\big)^{+}.$$

### 2.3 Host reliance reward (what TVC wraps)

The host scheme pays $i$ a credit $H_i^t = S_i(m^t, d^t) \ge 0$ that increases with reliance on $i$. Examples:
- popularity: $S_i = \beta\, \lvert\{j : d_j^t = i\}\rvert$;
- accuracy: $S_i = \gamma \sum_{j : d_j^t = i} (\theta_i^t - b_j^t)$;
- mixtures with share $\alpha = \beta/(\beta+\gamma)$;
- social-influence and peer-incentive rewards (L3).

Agent $i$'s reward under the host is $r_i^t = u_i^t + H_i^t$.

### 2.4 Truthful verified credit (TVC)

**Parameters:** tolerance $\tau \ge 0$, budget cap $\bar B > 0$, score weight $\eta > 0$, audit probability $p \in (0, 1]$, fine $\varphi \ge 0$, score width $w > 0$. The three amounts $\bar B$, $\eta$ and $\varphi$ are in units of $\bar v$.

**Events in round $t$:**

$$
\begin{aligned}
a_i^t &\sim \mathrm{Bernoulli}(p) && \text{audit, independent of everything else}\\
f_i^t &= \mathbb 1\big[\exists j : d_j^t = i\big] && \text{followed}\\
c_i^t &= \mathbb 1\big[m_i^t \neq \varnothing\big]\cdot \max\big(a_i^t, f_i^t\big) && \text{checked}\\
V_i^t &= \mathbb 1\Big[\Big\lvert \tfrac{m_i^t}{\theta_i^t} - 1 \Big\rvert \le \tau\Big] && \text{verified}\\
S(m, \theta) &= \exp\!\Big(-\frac{(\log m - \log \theta)^2}{2w^2}\Big) \in (0, 1], \quad S(\theta, \theta) = 1 && \text{score}
\end{aligned}
$$

**TVC credit:**

$$
C_i^t \;=\; \underbrace{\sum_{j : d_j^t = i} V_i^t \cdot \min\big(g_{ji}^t,\ \bar B\,\bar v\big)}_{\text{verified externality credit}}
\;+\; \underbrace{c_i^t\, \eta\, \bar v\; S\big(m_i^t, \theta_i^t\big)}_{\text{score bonus on checked claims}}
\;-\; \underbrace{a_i^t\,\big(1 - V_i^t\big)\, \mathbb 1\big[m_i^t \ne \varnothing\big]\, \varphi\, \bar v}_{\text{audit fine}}
$$

Silence earns $C_i^t = 0$. Under the wrapper, agent $i$'s reward is $r_i^t = u_i^t + C_i^t$. TVC **replaces** $H_i$. The code still adds the host's accuracy term $\gamma \cdot \text{gain}$, which is zero when $\alpha = 1$.

### 2.5 Assumptions

| ID | Assumption | Role |
| --- | --- | --- |
| **A1** Separability | Given the action profile, $u_i^t$ does not depend on $m_i^t$. Claims affect $i$'s reward only through $C_i$ and through receivers' future behaviour | P1, P3 |
| **A2** Unforgeable verification | On checked claims the designer learns $\theta_i$ exactly, and followers cannot misreport it | P1 (relax in P5) |
| **A3** Independent audits | $a_i^t$ is independent of $(m^t, d^t, h^t)$ | P1 |
| **A4** Lie separation | Every non-truthful claim misses the tolerance: $m \ne \theta \Rightarrow \lvert m/\theta - 1\rvert > \tau$. This holds for $\tau = 0$, or for the testbed's claim set $\{1, 1.1, 1.5, 2\}\cdot y$ with $\tau < 0.1$ | P1 (relax in P5 b) |
| **A5** Stage separability (optional) | Receiver rules do not depend on $i$'s past claims, or the horizon is myopic | P1 in the repeated game; otherwise see P1′ |

Limited liability is **not** assumed: the audit fine can make $C_i^t < 0$. A limited-liability variant would apply the fine as forfeiture of banked credit (open).

### 2.6 Solution concepts and objectives

- **Strict dominance (stage game).** Truthful reporting is *strictly dominant* if, for every receiver profile $\rho$ and every non-truthful claim $m$ (including silence),
  $$\mathbb E_a\big[C_i(y_i)\big] \;>\; \mathbb E_a\big[C_i(m)\big].$$
  The expectation is over audits only.
- **Learning guarantee.** For a learner with per-state regret $\mathrm{Reg}_i(T)$ against the fixed policy "always truthful", the fraction of non-truthful rounds is at most $\mathrm{Reg}_i(T)/(T\Delta)$, where $\Delta$ is the dominance margin (P3).
- **Designer's objective.** Maximise normalised welfare, which excludes transfers, subject to near-truthful reporting and a credit budget:
  $$\max\; W = \frac{1}{\bar v}\sum_{i,t} u_i^t \quad \text{s.t.} \quad \Pr\big[\kappa > 1 + \tau\big] \le \epsilon, \qquad \mathbb E\Big[\sum_{i,t} \big(C_i^t\big)^{+}\Big] \le \mathcal B .$$
- **Reported metrics:**
  - welfare $W$;
  - inflation rate: the share of claims with $\kappa > 1 + \tau$;
  - exploration rate;
  - credit per exploration;
  - outsider error $\mathbb E\lvert m/\theta - 1\rvert$ (C2).

### 2.7 Testbed instance (Gate A′)

| Quantity | Value |
| --- | --- |
| $n, K, T$ | 10, 50, 200 |
| $F$ | lognormal |
| $L$ | 3 |
| Claim set $\mathcal M$ | $\{\varnothing\} \cup \{1, 1.1, 1.5, 2\}\cdot y$ |
| $\sigma$ | 0 or 0.1 |
| $\tau$ | $0.05 + 2\sigma$ |
| $\bar B$ | 2, 8 or 32 |
| $\eta$ | 2 |
| $p$ | 0.2 |
| $\varphi$ | 8 |
| $w$ | $\max(\sigma, 0.05)$ |
| Receivers | naive (trust × claim), calibrated or persistent decoding |
| Learners | independent tabular Q-learning or softmax REINFORCE |
| Code | `tvc.py` (`tvc_credit`, `TVCParams`); `bandit.py` (this testbed); the pilot ran on `sim.py`, now removed |

**Note on the score.** The Gaussian kernel $S$ is maximised at the truth but is **not strictly proper in general** (see P5 d).

## 3. Results to prove

### Must prove (core of the paper)

**P1. Truth-telling is a strictly dominant strategy under TVC.**
- *Statement.* Let $\sigma = 0$ and assume A1–A4. Let $\bar s = \max_{m \ne \theta} S(m, \theta)$ be the best score any lie can earn. If
  $$p \cdot (\eta + \varphi) \;>\; \eta\, \bar s,$$
  then truthful reporting strictly beats every lie and silence, **for every receiver rule $\rho$**.
- *Proof idea.* Compare the worst case for truth with the best case for a lie (amounts in units of $\bar v$).
  - Truth: externality credit is $\ge 0$, and the claim is checked at least with probability $p$, so
    $$\mathbb E\big[C_i(\theta)\big] \;\ge\; p\,\eta .$$
  - A lie never passes the gate, so it earns no externality credit. Its score is at most $\bar s$, earned at most when followed, and it is fined with probability $p$:
    $$\mathbb E\big[C_i(m)\big] \;\le\; \eta\,\bar s - p\,\varphi .$$
  - Truth wins iff $p\eta > \eta\bar s - p\varphi$, which is the condition above.
  - Silence earns $0 < p\eta$.
- *Numbers* (testbed claim set $\{1, 1.1, 1.5, 2\}\cdot y$, so the closest lie is ×1.1; for a general lie rate $\kappa$, $\bar s = s(1+\tau) \approx 0.62$ at $\tau = 0.05$, and P1 still holds: $2.0 > 1.24$; see the matrix view). For a $\times 1.1$ lie with $w = 0.05$,
  $$\bar s = \exp\!\Big(-\frac{(\log 1.1)^2}{2 \cdot 0.05^2}\Big) \approx 0.16 .$$
  With the Gate A′ defaults ($p = 0.2$, $\eta = 2$, $\varphi = 8$): $p \cdot (\eta+\varphi) = 2.0 > \eta\bar s \approx 0.33$, so there is wide slack.
- *Why it matters.* Truth is best **whatever receivers do**. This is stronger than an equilibrium statement. It is robust to receivers who are themselves still learning, and it explains why TVC worked with naive, calibrated and persistent receivers alike. It also rewards reporting failures (silence < truth), which closes the strategic-silence channel (old H2′).
- *Difficulty.* Easy.

**P1, matrix view.** The same proof as a payoff matrix. Rows are the sender's claim and columns are the receiver's response. Entries are the sender's expected TVC credit, taken over audits only, in units of $\bar v$, with $g \ge 0$ the follower's gain.

A general lie reports $m = \kappa\,\theta$ with inflation rate $\kappa \neq 1$ ($\kappa > 1$ exaggerates, $\kappa < 1$ understates). By A4 it misses the tolerance ($\lvert\kappa - 1\rvert > \tau$, so $V = 0$), and its score is
$$s(\kappa) = S(\kappa\theta, \theta) = \exp\!\Big(-\frac{(\ln \kappa)^2}{2w^2}\Big) \in (0, 1), \qquad s(\kappa) = s(1/\kappa).$$

| Sender \ Receiver | Ignore | Follow |
| --- | --- | --- |
| Truth ($\kappa = 1$) | $p\,\eta$ | $\eta + \min(g, \bar B)$ |
| Lie (rate $\kappa$) | $p\,\big(\eta\, s(\kappa) - \varphi\big)$ | $\eta\, s(\kappa) - p\,\varphi$ |
| Silence | 0 | 0 |

*Numbers* (Gate A′ defaults: $p = 0.2$, $\eta = 2$, $\varphi = 8$, $w = 0.05$):

| $\kappa$ | $s(\kappa)$ | Ignore | Follow |
| --- | --- | --- | --- |
| 1 (truth) | 1 | **0.40** | ≥ **2.00** |
| 1.05 | 0.62 | −1.35 | **−0.36** |
| 1.1 | 0.16 | −1.54 | −1.28 |
| 1.5 | ≈ 0 | −1.60 | −1.60 |
| 2 | ≈ 0 | −1.60 | −1.60 |
| silence | – | 0 | 0 |

The most tempting lie is the **smallest admissible one**, because $s(\kappa)$ falls as $\kappa$ moves away from 1. So the best score any lie can earn is
$$\bar s = \sup_{\lvert\kappa - 1\rvert > \tau} s(\kappa) = s(1+\tau).$$
With $\tau = 0.05$: $\bar s \approx 0.62$.

How each cell arises:
- An ignored claim is checked only when it is audited (probability $p$).
- A followed claim is always checked.
- A lie fails the gate, so it earns no gain credit.
- A lie is fined $\varphi$ only when audited.

A receiver may respond differently to different claims, for example ignore the truth but follow a lie. Column-wise dominance is therefore not enough. Truth's **worst** cell must beat every other row's **best** cell:
$$\min(\text{truth row}) = p\,\eta \;>\; \sup_{\kappa}\max(\text{lie row}) = \eta\,\bar s - p\,\varphi \iff p \cdot (\eta + \varphi) > \eta\,\bar s, \qquad p\,\eta > 0 = \text{silence}.$$
- Truth's worst cell is *ignore*, because $\eta + g \ge \eta \ge p\eta$.
- For every $\kappa$, a lie's best cell is *follow*, because $\eta\,s(\kappa) \ge p\,\eta\,s(\kappa)$. The supremum over $\kappa$ is reached at $\kappa \to 1 + \tau$.
- A receiver that mixes (or learns) between the two columns yields a payoff between a row's minimum and maximum, so the comparison still holds.

∎ Here: $0.40 > -0.36$ (the worst case, $\kappa = 1.05$) and $0.40 > 0$.

*Small tolerance.* As $\tau \to 0$, the closest lie scores $\bar s \to 1$. P1 then needs $p\,\varphi > \eta\,(1 - p)$, i.e. $\varphi > \eta(1-p)/p$, which is $8$ at the defaults. So **$\varphi = 8$ is exactly the edge for arbitrarily small lies**. Either keep $\tau > 0$ (lies inside the tolerance count as truthful) or set $\varphi > \eta(1-p)/p$.

The same matrix for the alternatives shows why each part of TVC is needed (link to P2). Cells are (ignore, follow).

| Scheme | Truth | Lie | Silence | Verdict |
| --- | --- | --- | --- | --- |
| Popularity ($\beta$ per follower) | $(0, \beta)$ | $(0, \beta)$ | 0 | ✗ lie = truth; a lie wins if it draws more followers |
| Verification only (gain) | $(0, g)$ | $(0, 0)$ | 0 | ✗ ignored truth = silence = lie: indifference |
| Score only ($\eta S$ when followed) | $(0, \eta)$ | $(0, \eta\, s(\kappa))$ | 0 | ✗ ignored truth = silence; small lies still earn $\eta\, s(\kappa) > 0$ |
| **TVC** | $(p\eta,\ \eta + g)$ | $(p(\eta\, s(\kappa) - \varphi),\ \eta\, s(\kappa) - p\varphi)$ | 0 | ✓ worst truth > best lie (any $\kappa$), and > silence |

What each part contributes:
- the **audit** makes even ignored truth pay;
- the **score** pays for closeness to the truth;
- the **fine** pushes lies below silence;
- the **verified gain** pays for real help to followers.

*With an outside temptation (fake jams).* When a lie also earns the liar a private gain $G_1$ and harms the others by $H_1$, the lie's *follow* cell becomes $\eta\, s(\kappa) - p\varphi + (1-\lambda)G_1 - \lambda H_1 - c$. Truth then dominates iff
$$p \cdot (\eta + \varphi) > \eta\,\bar s + (1-\lambda)G_1 - \lambda H_1 - c,$$
which is the fine lower bound $\varphi_{\text{lo}}$.

**Corollary P1-T (outside temptation).** Suppose each lie also yields the sender a net private gain $\Gamma = (1-\lambda)G_1 - \lambda H_1 - c$, independent of audits (A3). Then truth strictly dominates every lie and silence for every receiver rule iff
$$p\cdot(\eta + \varphi) > \eta\,\bar s + \Gamma, \qquad\text{i.e.}\qquad \varphi > \varphi_{\text{lo}} = \frac{\eta\,\bar s + \Gamma}{p} - \eta.$$
*Proof:* add $\Gamma$ to the lie row's *follow* cell, its maximum, and repeat the minimum-against-maximum argument. ∎

In fake jams, a robot at a **free** doorway has no true jam claim to make, so the relevant comparison is lie against **silence** (payoff 0), not lie against truth. With a binary lie ($s \approx 0$), this gives $p\varphi > \Gamma$, i.e. $\varphi_{\text{lo}} = \Gamma/p$, which is the bound `calibrate()` prints. *Evidence* (FINDINGS §5c iii): at λ = 0 the predicted $\varphi_{\text{lo}} = 1.77$; the learned lie rate is 0.86 at $\varphi_{\text{lo}}$, 0.03 at $2\varphi_{\text{lo}}$ and 0.99 at $\varphi = 0$. Learners need a margin above the threshold (P3's $\Delta$). Run with `python -m tvc.fake_jams fine`.

**P1′. Repeated interaction (open).**
- *Issue.* When receivers carry memory, as the persistent receivers in the pilots do, today's claim changes tomorrow's follow rule. For example, under-reporting today lowers a receiver's decoding ratio, so the sender's future truthful claims look larger and attract more follows.
- *Target statement.* With discount factor $\delta$, truth remains dominant if the stage-game margin exceeds the discounted value of manipulating receiver state:
  $$p \cdot (\eta+\varphi) - \eta\,\bar s \;>\; \frac{\delta}{1-\delta}\,\Delta V, \qquad \Delta V \le n\,\bar B ,$$
  where $\Delta V$ bounds the extra future credit per round.
- *Alternatives.* Prove it for receivers without memory, or for receivers whose state updates only on verified claims.
- *Difficulty.* Moderate. **Check the pilot data for under-reporting:** the current inflation metric only counts over-claims.

**P2. Each part of TVC is necessary (minimality).**
- *Statement.* For each strict subset of {gate, externality credit, score, audit}, there exist parameters and a receiver rule under which truth is not strictly dominant.

| Variant | Counterexample | Pilot ($\sigma = 0$) |
| --- | --- | --- |
| Externality credit without the gate (`extern`) | A lie that is the only claim attracting a follower is paid $\min(L(\theta - b)^+, \bar B) > 0$; truth earns $0$ | 88% inflation |
| Popularity + gate only (VS, `vs`) | Unfollowed claims are unchecked, and followed lies forfeit to $0$ = truth's payoff when unfollowed. Truth is weakly but not strictly dominant, so learners drift (lemma L2) | 36–71% |
| Popularity + score (`score`) | The score is bounded ($\le \eta$) while popularity pays $\beta$ per follower. A lie that attracts $k$ follows earns $k\beta + \eta S$, which beats $p\eta$ once $k\beta > p\eta$ | 19% |
| Popularity + audit (`audit`) | Lying gains $\beta\,\Delta(\text{follows})$, unbounded in the number of followers, against an expected fine $p\varphi$. A smooth fine falls under Kartik's result (L1) | 31% |

- *Difficulty.* Easy to moderate. These are constructions, but they must match the code's rules exactly.

**P3. Learners converge to truthful reporting.**
- *Statement.* Under P1's conditions, let
  $$\Delta = p \cdot (\eta+\varphi) - \eta\,\bar s > 0 .$$
  Any learner with vanishing per-state regret plays non-truthful claims with frequency
  $$\frac{\#\{t \le T : m_i^t \ne y_i^t\}}{T} \;\le\; \frac{\mathrm{Reg}_i(T)}{T\,\Delta} \;\xrightarrow[T\to\infty]{}\; 0 .$$
  For softmax policy gradient on the claim choice, $\Pr[\text{truthful}] \to 1$ at the rate given by softmax-PG convergence results.
- *Proof idea.*
  - Dominance gives a gap of at least $\Delta$ in every state, against every receiver policy.
  - Regret against "always truthful" is at least $\Delta \times$ (number of lying rounds).
  - $\epsilon$-greedy Q-learning keeps lying at a rate below $\epsilon \times$ (share of non-truthful options) by construction.
- *Caveat.* This needs A1 and a stage-game reward. In the repeated setting it inherits the P1′ condition.
- *Why it matters.* MARL theory usually stops at equilibrium existence. This result covers the **learning dynamics** that the experiments actually run. It also accounts for the residual 0.3–0.4% inflation: exploration noise plus finite $T$.
- *Difficulty.* Easy given the citations.

**P4. Budget: TVC never pays more than the value it creates.**
- *Statement.*
  - Externality credit per follow is $\min(g_{ji}, \bar B\bar v) \le g_{ji}$. So
    $$\sum_{i,t}\ \sum_{j : d_j^t = i} V_i^t \min\big(g_{ji}^t, \bar B\bar v\big) \;\le\; \sum_{i,j,t} g_{ji}^t ,$$
    the total surplus created for followers. It is $0$ for follows that gained nothing.
  - Total payout is at most surplus $+\ \eta\bar v \times$ (number of checked claims) $-$ fines.
  - Popularity credit pays $\beta$ for every follow, including those with $g \le 0$.
- *Pilot link.* Credit per exploration was 1.5–2.7 under TVC against 3.3–5.9 for the other rules, at equal welfare.
- *Extension (old R1, after Groves).* In the two-agent game with $\bar B = \infty$, the sender explores iff
  $$c \;\le\; \mathbb E[\text{own value}] + \mathbb E\Big[\sum_j g_{ji}\Big],$$
  which is the planner's rule. A finite $\bar B$ gives a second-best rule, and the exploration incentive increases in $\bar B$.
- *Difficulty.* Bound easy; the planner comparison moderate.

### Should prove (supports the experiments)

**P5. Noise, tolerance $\tau$ and the score.**
- (a) An honest sender fails the check with probability
  $$q(\tau, \sigma) = \Pr\big[\lvert e^{\varepsilon} - 1\rvert > \tau\big] \approx 2\,\Phi(-\tau/\sigma).$$
- (b) Lies with $\lvert\kappa - 1\rvert < \tau$ pass the gate with probability close to $1 - q$. The dominance margin for small lies shrinks to about
  $$\eta\,\big(1 - \mathbb E[S]\big) + p\,\varphi\,\Delta\Pr[\text{caught}],$$
  so equilibrium inflation is roughly bounded by $\tau$.
  - This explains the 8–10% inflation at $\sigma = 0.1$: $\tau = 0.25$ lets the $\times 1.1$ lie through.
- (c) The welfare-optimal $\tau^\star$ increases in $\sigma$. It trades honest forfeits, which lower exploration, against tolerated lies. Likely numeric.
- (d) **Score properness.** The Gaussian kernel score is maximised at the posterior mode. It is truthful under noise only for symmetric, unimodal posteriors on $\log v$. Either prove it under that condition, or switch the code to a strictly proper rule (log score, or quadratic on a discretised value) and re-run.
- *Difficulty.* (a)–(b) easy; (c) numeric; (d) easy, but it may require a code change.

**P6. Two levers (old T5).**
- *Statement.* Under truthful reporting, a sender explores iff
  $$c \;\le\; \mathbb E[\text{own value}] + \mathbb E[\text{credit per exploration}].$$
  The threshold depends on how much credit is paid per useful discovery, not on whether it is labelled popularity or accuracy. Inflation depends on the form of credit ($\beta/\gamma$ for the host scheme; $\approx 0$ under TVC).
- *Pilot link.* Exploration tracked credit per exploration across $\alpha$. TVC's slightly lower exploration matches its lower credit per exploration.
- *Difficulty.* Moderate.

### Background lemmas (cite; little new work)

- **L1 (after Kartik 2009). Smooth penalties cannot stop lying.** For any continuous fine $\varphi(d)$ with $\varphi(0) = 0$ and any reliance reward $\beta > 0$, truth is not an equilibrium. A claim just over the follow threshold wins $\beta$, while the fine and accuracy loss go to $0$. With a quadratic fine $\lambda d^2$, inflation is about
  $$d^\star \approx \sqrt{\beta/\lambda}:$$
  it shrinks with $\lambda$ but never reaches $0$. **Not new.** Kartik covers continuous actions; ours adapts it to binary follow decisions. This is why TVC uses a pass/fail gate plus fixed fines.
- **L2 (after Townsend 1979; Ben-Porath, Dekel & Lipman 2014). Verification alone gives only weak truthfulness.** Gating reliance credit on verification removes the gain from lying (old T2), and works under limited liability where accuracy pay fails (old T3). But unfollowed claims earn $0$ either way, so truth is not *strictly* dominant. This explains Gate A and motivates the score and audits.
- **L3 (old T4). Wrapper generality.** Any reliance reward $S_i \ge 0$ that increases in others' reliance on $i$'s claims can be replaced by TVC's gated, capped form. P1 then holds under A1 and A2. This is what makes TVC a drop-in wrapper for social-influence rewards (Jaques et al. 2019), LIO and PIMAEX.

### Conjectures (for experiments, not proofs)

- **C1. Hype hurts when senders inflate by different amounts** (after Frankel & Kartik 2019; Fischer & Verrecchia 2000). If receivers cannot learn each sender's inflation $\kappa_i$, hype reduces how informative claims are and lowers welfare, and TVC restores it.
  - *Test:* heterogeneous caps (half the agents at $\times 1.1$, half at $\times 2$), or receivers that cannot track senders individually.
- **C2. Outsider value.** A naive outside reader's error, $\mathbb E\lvert m/\theta - 1\rvert$, is about $0$ under TVC and large under every host scheme, at equal in-system welfare.
  - *Test:* add the metric. Nearly true by construction, so present it as measurement rather than theory.

### Dropped or deferred

- **Old T6** (hype destroys popularity credit with rational receivers; VS restores exploration): not observed. Persistent receivers decode hype instead of ignoring it, which is Kartik's "language inflation" outcome. Dropped.
- **Old T8** (credit cannot fully fix free-riding) and **old T9** (common-value arms bring back strategic silence): future work.

### Priority and effort

| Priority | Results | Rough effort |
| --- | --- | --- |
| Must | P1, P2, P3, P4 (bound) | 1–2 weeks |
| Should | P5 (a, b, d), P6, P1′ | 2–3 weeks |
| Optional | P4 planner comparison, P5 (c) numeric | ~1 week |
| Experiments | C1, C2 | Next session |

If a proof stalls, solve the two-agent game numerically on a discretised value space and report a numerical proposition.

## 4. Positioning and novelty

Each ingredient is classic:
- verification-contingent rewards (Townsend; Ben-Porath, Dekel & Lipman);
- the failure of smooth penalties (Kartik);
- proper scoring (Gneiting & Raftery);
- externality pricing (Groves);
- dominated strategies vanishing under no-regret learning.

**P1 on its own may be called obvious.** The contribution lies in what those papers do not cover:

1. **Verification comes from peers' reliance.** Checks happen because someone acted on the claim, plus cheap random audits. There is no principal paying to verify every claim.
2. **Minimality (P2).** All three parts are needed, and each fails alone in a different way (L1, L2, and the unbounded reliance reward).
3. **Learning agents (P3).** The guarantee holds for the dynamics MARL actually runs, not only at equilibrium, and it holds whatever receivers do.
4. **Budget (P4).** Pay is bounded by the value created (the $\bar B$ cap and verified gain). The pilot's "same welfare at half the budget" was **not** confirmed on a matched-budget frontier, so P4 is a safety bound, not an efficiency claim.
5. **A wrapper for MARL reward schemes (L3).** Applies to social-influence, LIO and PIMAEX-style rewards.

**Positioning sentence.** *"Reliance-based rewards in multi-agent learning pay agents for moving others, which can be gamed by overclaiming. We combine verification-contingent credit, a score bonus and random audits into a wrapper under which truthful reporting is strictly dominant for any receiver behaviour. We show that each component is necessary, that no-regret learners converge to truthful reporting, and that payments never exceed the value created."*

Novelty checks already done (via NotebookLM on the source papers):

| Risk | Finding | Decision |
| --- | --- | --- |
| L1 ≈ Kartik (2009) | Confirmed: no fully truthful equilibrium with smooth costs; audit-imposed costs covered; continuous actions only | Background lemma |
| L2 ≈ verification literature | Confirmed: forfeiture on a failed check, limited liability, many agents; **no repeated or learning setting** | Background lemma; import |

**Still open:**
- Does any paper combine externality credit, scoring rules and audits for multi-agent *learning*? Search terms: "Groves mechanism multi-agent reinforcement learning", "proper scoring rule communication MARL", "truthful reporting incentive multi-agent learning".
- Do Ben-Porath et al. allow noisy verification? This affects the novelty of P5.

## 5. Reading list by result

| Result | Read | Why |
| --- | --- | --- |
| **P1, L2** | **Townsend (1979)**, "Optimal Contracts and Competitive Markets with Costly State Verification", *JET* | Payments contingent on verification |
| | **Ben-Porath, Dekel & Lipman (2014)**, "Optimal Allocation with Costly Verification", *AER* | Forfeit on a failed check; closest formal ancestor |
| | Börgers (2015), *An Introduction to the Theory of Mechanism Design* | Dominant-strategy incentive compatibility; proof toolkit |
| **P1, P5 (d)** | **Gneiting & Raftery (2007)**, "Strictly Proper Scoring Rules, Prediction, and Estimation", *JASA* | Score properness; choose a strictly proper $S$ |
| **P2, L1** | **Kartik (2009)**, "Strategic Communication with Lying Costs", *RES* | Smooth penalties leave inflation |
| | Kartik, Ottaviani & Squintani (2007), "Credulity, Lies, and Costly Talk", *JET* | Inflation with naive receivers |
| | Innes (1990), "Limited Liability and Incentive Contracting with Ex-ante Action Choices", *JET* | Limited liability |
| **P3** | Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and Games* | No-regret learning; dominated actions vanish |
| | Hart & Mas-Colell (2000), "A Simple Adaptive Procedure Leading to Correlated Equilibrium", *Econometrica* | Regret dynamics in games |
| | Mei, Xiao, Szepesvári & Schuurmans (2020), "On the Global Convergence Rates of Softmax Policy Gradient Methods", *ICML* | Convergence rate for the PG learner |
| **P4** | Groves (1973), "Incentives in Teams", *Econometrica*; Clarke (1971), *Public Choice*; Vickrey (1961), *J. Finance* | Externality pricing |
| | Holmström (1982), "Moral Hazard in Teams", *Bell J. Econ.* | Budget limits |
| | Wolpert & Tumer (2001), "Optimal Payoff Functions for Members of Collectives", *Advances in Complex Systems*; Foerster et al. (2018), COMA, *AAAI* | Difference rewards: the MARL counterpart of externality credit |
| **P4, P6** | Bolton & Harris (1999), *Econometrica*; Keller, Rady & Cripps (2005), *Econometrica* | Free-riding and the planner benchmark for exploration |
| | Kremer, Mansour & Perry (2014), *JPE*; Mansour, Slivkins & Syrgkanis (2015); Slivkins (2019), *Introduction to Multi-Armed Bandits*, incentivised-exploration chapter | Incentivised exploration |
| **C1** | **Frankel & Kartik (2019)**, "Muddled Information", *JPE* | Different inflation across senders destroys information |
| | Fischer & Verrecchia (2000), "Reporting Bias", *Accounting Review* | Bias is harmless only when its size is known |
| | Chan, Hao & Suen (2007), "A Signaling Theory of Grade Inflation", *IER* | Inflation that varies across senders |
| **C2** | Ottaviani & Squintani (2006), "Naive Audience and Communication Bias", *JET*; Malmendier & Shanthikumar (2007), *JFE* | Harm to naive or outside readers |
| **L3 (MARL hosts)** | Jaques et al. (2019), social influence, *ICML*; Yang et al. (2020), LIO, *NeurIPS*; Kölle et al. (2025), PIMAEX, *ICAART* | Reward schemes the wrapper attaches to |
| **Setting** | Heidhues, Rady & Strack (2015), *JET*; Renault, Solan & Vieille (arXiv 2512.06180); Rosenberg, Solan & Vieille (2007), *Econometrica*; Akcigit & Liu (2016), *JEEA*; Vial, Shakkottai & Srikant (2021), robust multi-agent bandits | Private payoffs, strategic silence, dishonest recommenders |

### Suggested reading order

1. **Gneiting & Raftery (2007).** Fix the scoring rule first (P5 d).
2. **Kartik (2009).** L1, and the language-inflation account of the pilot results.
3. **Townsend (1979)**, then **Ben-Porath, Dekel & Lipman (2014)**. L2 and P1.
4. **Cesa-Bianchi & Lugosi (2006)**, chapters on regret and games, plus **Mei et al. (2020)**. P3.
5. **Groves (1973)** and **Wolpert & Tumer (2001)**. P4.
6. **Frankel & Kartik (2019).** C1 experiment design.

## 6. Next steps

1. Write P1 and P2 formally against the exact rules in `tvc.py`. Check the counterexamples numerically with `tvc_credit`.
2. Decide on the score: prove P5 (d) for the Gaussian kernel, or switch to the log or quadratic score and re-run Gate A′.
3. Add an under-reporting metric (P1′) and the outsider-error metric (C2) to `bandit.py`.
4. Run the C1 heterogeneous-sender experiment.
5. Run a citation-verification pass on Section 5, and the literature search in Section 4.







