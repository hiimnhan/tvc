"""Intuition figures for TVC: one sender, one receiver, one round.

Writes to <repo>/figures/:
  payoff_group.png        sender's pay (popularity vs TVC) and group value, for each claim size
  honesty_every_case.png  truth vs best lie vs silence under TVC, for every receiver rule
Run: python visualisation.py
Also holds the fake-jams figures and video (plot_sweep, plot_scale, plot_listen, plot_hype, plot_hype_compare,
make_video), called from the modes of fake_jams.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import replace

from tvc.fake_jams import LOGS, SHOW, make_policy, run_episode
from tvc.tvc import TVCParams, tvc_credit

OUT = Path(__file__).resolve().parents[2] / "figures"
B = 1.0  # follower's own best value
BETA = 2.0  # popularity credit per follow
MEAN_V = 1.0
LIES = [0.9, 1.1, 1.5, 2.0]
RULES = {  # receiver rule: does R follow claim m, given the true value th?
    "Never\nfollows": lambda m, th: False,
    "Naive\n(m > b)": lambda m, th: m > B,
    "Always\nfollows": lambda m, th: True,
    "Adversarial\n(lies only)": lambda m, th: not np.isclose(m, th),
}
FINDS = [(2.0, "good"), (0.8, "low-value")]


def tvc_pay(theta, m, follows, p=TVCParams()):
    """TVC credit for one claim m, expected over the audit coin."""
    rng = np.random.default_rng(0)
    s = np.array([0]) if follows else np.array([], int)
    args = (
        np.array([m]),
        np.ones(1, bool),
        np.array([theta]),
        s,
        np.full(len(s), theta),
        np.full(len(s), B),
        MEAN_V,
    )
    no_audit = tvc_credit(rng, *args, p=TVCParams(**{**p.__dict__, "p_audit": 0.0}))[0]
    audit = tvc_credit(rng, *args, p=TVCParams(**{**p.__dict__, "p_audit": 1.0}))[0]
    return (1 - p.p_audit) * no_audit + p.p_audit * audit


def pop_pay(theta, m, follows):
    return BETA * follows


def group_value(theta, m):
    """What the group actually gains: follower's payoff change over the lock-in (transfers excluded)."""
    return TVCParams().lock_in * (theta - B) * (m > B)


def plot_payoff_group(path):
    ks = np.linspace(0.5, 2.5, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (theta, find) in zip(axes, FINDS):
        m = ks * theta
        grp = group_value(theta, m)
        ax.fill_between(
            ks,
            grp,
            0,
            where=grp < 0,
            color="tab:green",
            alpha=0.15,
            label="Group loses (follower misled)",
        )
        ax.axhline(0, color="grey", lw=1, ls=":")
        ax.plot(
            ks, grp, color="tab:green", lw=3, label="Group value (follower's real gain)"
        )
        ax.plot(
            ks,
            pop_pay(theta, m, m > B),
            color="tab:red",
            lw=2,
            label="Sender's pay: popularity",
        )
        ax.plot(
            ks,
            [tvc_pay(theta, x, x > B) for x in m],
            color="tab:blue",
            lw=2,
            label="Sender's pay: TVC (expected)",
        )
        ax.axvline(1.0, color="k", lw=1, ls="--")
        ax.set_title(
            f"{find.capitalize()} find: true value {theta} (follower's best = {B})",
            fontsize=10,
        )
        ax.set_xlabel("Claim ÷ true value  (1 = honest)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value / credit")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Sender's pay vs what the group actually gains, for each claim (dashed = truth)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def every_case():
    """(labels, {rule_name: [(truth, best_lie, silence), ...]}) over receiver rules x finds."""
    labels, res = [], {"TVC": [], "Popularity": []}
    for theta, find in FINDS:
        for rname, rule in RULES.items():
            labels.append(f"{rname}\n[{find} find]")
            for name, pay in [("TVC", tvc_pay), ("Popularity", pop_pay)]:
                truth = pay(theta, theta, rule(theta, theta))
                best_lie = max(
                    pay(theta, k * theta, rule(k * theta, theta)) for k in LIES
                )
                res[name].append((truth, best_lie, 0.0))
    return labels, res


def plot_every_case(path):
    """Truth vs best lie vs silence under TVC only, for every receiver rule and find."""
    labels, res = every_case()
    x, w = np.arange(len(labels)), 0.27
    fig, ax = plt.subplots(figsize=(13, 4.6))
    t, l, s = map(np.array, zip(*res["TVC"]))
    ax.bar(x - w, t, w, color="tab:blue", label="Tell the truth")
    ax.bar(x, l, w, color="tab:red", label="Best possible lie")
    ax.bar(x + w, s, w, color="grey", label="Stay silent")
    for xi, (ti, li) in enumerate(zip(t, l)):
        mark = (
            "✓ truth wins"
            if ti > li + 1e-9
            else ("tie" if np.isclose(ti, li) else "✗ lie wins")
        )
        ax.text(
            xi,
            max(ti, li, 0) + 0.25,
            mark,
            ha="center",
            fontsize=8,
            color="tab:green" if mark.startswith("✓") else "tab:red",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(len(RULES) - 0.5, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Sender's expected credit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-2.3, 6.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- fake-jams figures and video

NAMES = dict(tvc="TVC")  # display names for rules in figures


def read_table(path) -> list[dict[str, str]]:
    """Rows of a markdown table written by sweep / scale mode, as {header: cell}."""
    lines = [
        l
        for l in Path(path).read_text().splitlines()
        if l.startswith("| ") and not l.startswith("| ---")
    ]
    head = [h.strip() for h in lines[0].strip("|").split("|")]
    return [
        dict(zip(head, (c.strip() for c in l.strip("|").split("|")))) for l in lines[1:]
    ]


def mean_ci(cell) -> tuple[float, float]:
    m, h = cell.split("±")
    return float(m), float(h)


def plot_sweep(out, values, xlabel, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(
        none="grey", popularity="tab:red", accuracy="tab:orange", tvc="tab:blue"
    )
    fig, axes = plt.subplots(1, len(SHOW), figsize=(4.2 * len(SHOW), 3.8))
    for ax, k in zip(axes, SHOW):
        for rule, rows in out.items():
            ax.errorbar(
                values,
                [r[k] for r in rows],
                yerr=[r.get(k + "_ci", 0) for r in rows],
                fmt="o-",
                capsize=3,
                color=colors.get(rule),
                label=NAMES.get(rule, rule),
            )
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_audit(rows, ps, p_star, path):
    """Audit-rate sweep: rows[arm][i][metric] = (mean, ci) at ps[i]; p_star = predicted lying threshold of the fixed fine."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("lie_rate", "lie rate"),
        ("pogema_avg_throughput", "throughput"),
        ("audits", "audits per step"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (k, title) in zip(axes, panels):
        for (arm, rs), c in zip(rows.items(), ("tab:blue", "tab:green")):
            ax.errorbar(ps, [r[k][0] for r in rs], yerr=[r[k][1] for r in rs], fmt="o-", capsize=3, color=c, label=arm)
        ax.axvline(p_star, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("audit rate p")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scale(rows, layouts, rules, path):
    """Lie rate and POGEMA throughput vs agents; rows[(rooms, agents, rule)][metric] = (mean, ci)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(none="grey", popularity="tab:red", tvc="tab:blue")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, k in zip(axes, ("lie_rate", "pogema_avg_throughput")):
        for rooms, style in ((3, "o-"), (4, "s--")):
            for r in rules:
                pts = [(a, rows[(rr, a, r)][k]) for rr, a in layouts if rr == rooms]
                ax.errorbar(
                    [a for a, _ in pts],
                    [m for _, (m, _) in pts],
                    yerr=[h for _, (_, h) in pts],
                    fmt=style,
                    capsize=3,
                    color=colors[r],
                    label=f"{NAMES.get(r, r)} ({rooms}x{rooms})",
                )
        ax.set(xlabel="agents", title=k.replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_listen(out, envs, rhos, path):
    """Throughput (left) and lie rate (right) against rho (receivers' chance of acting on a claim), none vs TVC,
    one row per env."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("throughput", "Throughput (deliveries per robot-step)"),
        ("lie_rate", "Lie rate"),
    ]
    fig, axes = plt.subplots(len(envs), 2, figsize=(9, 3.3 * len(envs)), squeeze=False)
    for row, env in enumerate(envs):
        for col, (k, label) in enumerate(metrics):
            ax = axes[row, col]
            for rule, color, name in [
                ("none", "grey", "No credit"),
                ("tvc", "tab:blue", "TVC"),
            ]:
                ms, cs = zip(*[out[f"{env}, rho {r:g}: {rule}"][k] for r in rhos])
                ax.errorbar(
                    rhos, ms, yerr=cs, fmt="o-", capsize=3, color=color, label=name
                )
            ax.set_ylabel(label)
            ax.set_xlabel(r"$\rho$: chance a receiver acts on a claim")
            ax.grid(alpha=0.3)
            if len(envs) > 1:
                ax.set_title(env.capitalize())
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hype(rows, tag=""):
    """Group, honest and lying robots' deliveries and the share of misled detours against the number of lying
    robots (rows from hype_sweep)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["k"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for key, color in (
        ("group", "black"),
        ("honest", "tab:blue"),
        ("hypers", "tab:red"),
    ):
        a1.errorbar(
            x,
            [r[key][0] for r in rows],
            yerr=[r[key][1] for r in rows],
            fmt="o-",
            capsize=3,
            color=color,
            label=key,
        )
    a1.set(
        xlabel="number of hype robots",
        ylabel="deliveries per robot per episode",
        title="hype hurts the group",
    )
    a1.legend(fontsize=8)
    a2.errorbar(
        x,
        [r["misled"][0] for r in rows],
        yerr=[r["misled"][1] for r in rows],
        fmt="o-",
        capsize=3,
        color="tab:red",
    )
    a2.set(
        xlabel="number of hype robots",
        ylabel="share of detours that were for fake jams",
        title="followers misled",
    )
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"fake_jams_hype{tag}.png", dpi=150)
    plt.close(fig)



def plot_hype_compare(path=None) -> None:
    """One figure from the two hype tables (naive and sceptical receivers): deliveries by group, honest and
    lying robots for each receiver type, and the share of detours that were for fake jams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read(tag):
        rows = {}
        for line in (LOGS / f"fake_jams_hype{tag}.md").read_text().splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c[0].isdigit():
                rows[int(c[0])] = {
                    k: tuple(float(v) for v in x.split("±"))
                    for k, x in zip(("group", "honest", "hypers", "misled"), c[1:])
                }
        return rows

    data = {"Naive receivers": read(""), "Sceptical receivers": read("_sceptical")}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (title, rows) in zip(axes[:2], data.items()):
        ks = sorted(rows)
        for key, color, label in (
            ("group", "black", "group"),
            ("honest", "tab:blue", "honest robots"),
            ("hypers", "tab:red", "lying robots"),
        ):
            ax.errorbar(
                ks,
                [rows[k][key][0] for k in ks],
                yerr=[rows[k][key][1] for k in ks],
                fmt="o-",
                ms=4,
                capsize=2,
                color=color,
                label=label,
            )
        ax.set(
            xlabel="number of lying robots",
            title=title,
            ylim=(0.8, 9.6),
            xticks=[0, 3, 6, 9, 12],
        )
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("deliveries per robot per episode")
    axes[0].legend(fontsize=8)
    for (title, rows), color in zip(data.items(), ("tab:red", "tab:purple")):
        ks = sorted(rows)
        axes[2].errorbar(
            ks,
            [rows[k]["misled"][0] for k in ks],
            yerr=[rows[k]["misled"][1] for k in ks],
            fmt="o-",
            ms=4,
            capsize=2,
            color=color,
            label=title,
        )
    axes[2].set(
        xlabel="number of lying robots",
        ylabel="share of detours for fake jams",
        title="Followers misled",
        xticks=[0, 3, 6, 9, 12],
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path or OUT / "fake_jams_hype_compare.png", dpi=200)
    plt.close(fig)


def make_video(cfg, path, seed=7, fps=12) -> None:
    """Side-by-side MP4 of one episode, same seed: everyone reports real jams vs everyone hypes.
    Grey = walls, dots = robots, rings = their targets, orange square = real jam,
    green square = active true claim, red X = active fake claim."""
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    worlds = {
        "honest: robots report real jams": lambda i, s: int(s in (1, 3)),
        "hype: robots post fake jams on their own route": lambda i, s: int(s == 2),
    }
    recs = {}
    for title, forced in worlds.items():
        recs[title] = {}
        run_episode(
            replace(cfg, rule="none"),
            np.random.default_rng(seed),
            make_policy(cfg),
            np.ones((cfg.agents, cfg.agents)),
            learn=False,
            forced=forced,
            record=recs[title],
        )

    colors = plt.cm.tab20(np.arange(cfg.agents) % 20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    artists = []
    for ax, (title, rec) in zip(axes, recs.items()):
        ax.imshow(rec["walls"], cmap="Greys", vmin=0, vmax=1.6)
        ax.set_xticks([]), ax.set_yticks([])
        ys, xs = np.nonzero(rec["walls"])  # crop the empty padding around the map
        (
            ax.set_xlim(xs.min() - 0.5, xs.max() + 0.5),
            ax.set_ylim(ys.max() + 0.5, ys.min() - 0.5),
        )
        dy, dx = np.array(rec["doorways"]).T
        jam = ax.scatter([], [], s=150, marker="s", color="tab:orange", alpha=0.6)
        true_c = ax.scatter(
            [],
            [],
            s=150,
            marker="s",
            facecolors="none",
            edgecolors="tab:green",
            linewidths=2,
        )
        fake_c = ax.scatter([], [], s=120, marker="x", color="red", linewidths=2.5)
        tgt = ax.scatter(
            np.zeros(cfg.agents),
            np.zeros(cfg.agents),
            s=60,
            facecolors="none",
            edgecolors=colors,
            linewidths=1.5,
        )
        bot = ax.scatter(
            np.zeros(cfg.agents),
            np.zeros(cfg.agents),
            s=70,
            color=colors,
            edgecolors="black",
            linewidths=0.5,
        )
        txt = ax.set_title("", fontsize=10)
        artists.append((rec, title, dy, dx, jam, true_c, fake_c, tgt, bot, txt))

    def yx(points):  # (y, x) -> scatter offsets (x, y)
        return np.array([(x, y) for y, x in points]).reshape(-1, 2)

    def update(t):
        out = []
        for rec, title, dy, dx, jam, true_c, fake_c, tgt, bot, txt in artists:
            f = rec["frames"][t]
            jam.set_offsets(np.c_[dx[f["jammed"]], dy[f["jammed"]]].reshape(-1, 2))
            true_d = sorted({d for d, ok in f["claims"] if ok})
            fake_d = sorted({d for d, ok in f["claims"] if not ok})
            true_c.set_offsets(np.c_[dx[true_d], dy[true_d]].reshape(-1, 2))
            fake_c.set_offsets(np.c_[dx[fake_d], dy[fake_d]].reshape(-1, 2))
            tgt.set_offsets(yx(f["targets"]))
            bot.set_offsets(yx(f["agents"]))
            txt.set_text(
                f"{title}\nstep {t + 1}/{len(rec['frames'])}   deliveries {f['delivered']:.0f}   "
                f"robots avoiding a doorway {f['believers']}"
            )
            out += [jam, true_c, fake_c, tgt, bot, txt]
        return out

    fig.text(
        0.5,
        0.02,
        "dots = robots, rings = their targets   |   orange square = real jam   |   "
        "green square = true claim   |   red X = fake claim (free doorway)",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    anim = FuncAnimation(
        fig, update, frames=len(next(iter(recs.values()))["frames"]), blit=False
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    for title, rec in recs.items():
        print(f"{title}: {rec['frames'][-1]['delivered']:.0f} deliveries")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plot_payoff_group(OUT / "payoff_group.png")
    plot_every_case(OUT / "honesty_every_case.png")

    _, res = (
        every_case()
    )  # P1 check: truth strictly beats every lie and silence under TVC, not under popularity
    assert all(t > l and t > s for t, l, s in res["TVC"])
    assert any(l > t for t, l, _ in res["Popularity"])
    print(f"figures written to {OUT}; TVC truth-dominance check passed")
