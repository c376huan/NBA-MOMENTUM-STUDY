"""
NBA Momentum Study - Step 4: is momentum real?

Question: after a "momentum moment" (a scoring run, a dunk, a block, a steal
that leads to a score, an and-one, a timeout during an opponent's run), does
the team do better over the next few possessions than the game situation
alone would predict?

Everything is measured as a POINTS-PER-POSSESSION MARGIN over the next N
possessions (N = 4, 6, 10; both teams' possessions counted), shown as points
per 100 possessions - the same unit as a team's "net rating". +5 would be a
big effect (roughly the gap between an average team and a very good one).

Three parts:

 A. Event study + adjustment.
    "before"   = the team's margin over the N possessions up to the moment.
    "after"    = its raw margin over the N possessions after the moment.
    "adjusted" = after, minus what we'd EXPECT in that situation anyway.
    The expectation comes from a regression fitted on every possession in
    the data. It knows: how good the two teams are, the score margin (leading
    teams ease off), the period and clock, home court, and how the next
    possession starts (after a made basket the other team inbounds against a
    set defense and scores less than after a miss or steal - that is not
    momentum, it's just how basketball works, so we remove it).
    Why bother: good teams go on more runs AND keep outscoring opponents, so
    the raw "after" number is positive even with zero momentum.

 B. Timeouts. Among moments when a team has just given up 6+ unanswered
    points, compare what happens next when it calls a timeout vs when it
    doesn't. Comparing run-vs-run matters: runs end on their own (regression
    to the mean), so "the run stopped after the timeout" proves nothing.

 C. Shuffle test. If possessions were independent coin flips (no momentum at
    all), how many 8-0 / 10-0 / 12-0 runs would we see just by luck? We
    shuffle each team's possession results within each game many times and
    count. Real games having clearly more runs than the shuffles = streakiness.

Uncertainty: the independent unit is the GAME, not the possession, so all
standard errors are "clustered by game". With ~20 tests some will look
significant by luck, so p-values are also shown after the Holm correction.

Usage:
    python momentum.py                      # real data
    python momentum.py --shuffles 500       # more shuffles (slower, smoother)
    python momentum.py --db data/fake_nba.db
"""

import sqlite3
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
N_SHUFFLES = arg("--shuffles", 200)
WINDOWS = (4, 6, 10)        # how many possessions to look ahead
RUN_SIZES = (8, 12)         # unanswered points that count as "a run"
TIMEOUT_RUN = 6             # timeouts: opponent has scored this many unanswered
rng = np.random.default_rng(0)

# ============================================================== load data
con = sqlite3.connect(DB)
poss = pd.read_sql_query("SELECT * FROM poss ORDER BY game_id, poss_num", con)
events = pd.read_sql_query(
    """SELECT game_id, poss_num, period, seconds_left, eventmsgtype AS etype, pts,
              homedescription AS home_txt, visitordescription AS away_txt
       FROM events WHERE poss_num IS NOT NULL OR eventmsgtype = 9""", con)
con.close()

n = len(poss)
game = poss["game_id"].to_numpy()
period = poss["period"].to_numpy()
sign = np.where(poss["is_home"] == 1, 1, -1)          # +1 if the home team has the ball
ok = poss["points_ok"].to_numpy().astype(bool)
garbage = poss["garbage"].to_numpy().astype(bool)
pts_off = np.where(ok, poss["points"], 0.0)
pts_def = np.where(ok, poss["def_points"], 0.0)
home_pts = np.where(sign == 1, pts_off, pts_def)
away_pts = np.where(sign == 1, pts_def, pts_off)
net = home_pts - away_pts                              # home team's net points, per possession


def csum(x):
    """Running total with a leading 0, so sum(x[a:b]) = c[b] - c[a]."""
    return np.concatenate([[0.0], np.cumsum(x)])


C = {name: csum(x) for name, x in dict(
    hp=home_pts, ap=away_pts, hn=(sign == 1), an=(sign == -1), bad=~ok, garb=garbage).items()}


def window_margin(a, b):
    """Home team's points-per-possession margin (x100) over possessions a..b-1.

    a and b are arrays (one window per row). Returns NaN when a window is not
    usable: it runs off the game, crosses into another period (end-of-quarter
    breaks and heaves), touches garbage time, or contains a possession whose
    points failed the data check.
    """
    out = np.full(len(a), np.nan)
    good = (a >= 0) & (b <= n)
    a, b = a[good], b[good]
    hn, an = C["hn"][b] - C["hn"][a], C["an"][b] - C["an"][a]
    valid = ((game[a] == game[b - 1]) & (period[a] == period[b - 1])
             & (C["bad"][b] - C["bad"][a] == 0) & (C["garb"][b] - C["garb"][a] == 0)
             & (hn > 0) & (an > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 100 * ((C["hp"][b] - C["hp"][a]) / hn - (C["ap"][b] - C["ap"][a]) / an)
    out[good] = np.where(valid, m, np.nan)
    return out


idx = np.arange(n)
after = {N: window_margin(idx + 1, idx + 1 + N) for N in WINDOWS}    # the N possessions after i
before = {N: window_margin(idx + 1 - N, idx + 1) for N in WINDOWS}   # the N up to and including i


def ols_cluster(X, y, groups):
    """Ordinary least squares, with standard errors clustered by `groups`.

    Possessions in the same game are not independent, so we add up each
    game's contribution before measuring the spread (the "sandwich" formula).
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    scores = X * (y - X @ beta)[:, None]
    per_game = pd.DataFrame(scores).groupby(groups).sum().to_numpy()
    G = len(per_game)
    cov = XtX_inv @ (per_game.T @ per_game) @ XtX_inv * G / (G - 1)
    return beta, np.sqrt(np.diag(cov))


def dummies(values):
    """One 0/1 column per distinct value."""
    return pd.get_dummies(values).to_numpy(dtype=float)


# =================================================== what to expect anyway
# Team strength = season net points per possession, leaving out the game
# itself (so a team's big run tonight can't make it look "strong" tonight).
use = ok & ~garbage
per_game = pd.DataFrame({"game_id": game, "hp": home_pts * use, "ap": away_pts * use,
                         "hn": (sign == 1) & use, "an": (sign == -1) & use}).groupby("game_id").sum()
info = poss.groupby("game_id")[["season", "home", "away"]].first()
tg = pd.concat([   # one row per team per game: points/possessions for and against
    pd.DataFrame({"season": info["season"], "team": info["home"], "pf": per_game["hp"],
                  "nf": per_game["hn"], "pa": per_game["ap"], "na": per_game["an"]}),
    pd.DataFrame({"season": info["season"], "team": info["away"], "pf": per_game["ap"],
                  "nf": per_game["an"], "pa": per_game["hp"], "na": per_game["hn"]})]).reset_index()
rest = tg.groupby(["season", "team"])[["pf", "nf", "pa", "na"]].transform("sum") - tg[["pf", "nf", "pa", "na"]]
tg["strength"] = 100 * (rest["pf"] / rest["nf"] - rest["pa"] / rest["na"])
strength = tg.set_index(["game_id", "team"])["strength"]
home_strength = strength.reindex(pd.MultiIndex.from_frame(poss[["game_id", "home"]])).to_numpy()
away_strength = strength.reindex(pd.MultiIndex.from_frame(poss[["game_id", "away"]])).to_numpy()
strength_diff = np.nan_to_num(home_strength - away_strength)        # home minus away

# Situation at the start of the NEXT possession (j = i + 1), home team's view.
j = np.minimum(idx + 1, n - 1)
home_margin = (poss["start_margin"].to_numpy() * sign)[j]
margin_bucket = np.digitize(home_margin, [-15.5, -10.5, -5.5, -2.5, -0.5, 0.5, 2.5, 5.5, 10.5, 15.5])
start_type = poss["starttype"].fillna("?").to_numpy()[j]
X_base = np.column_stack([
    np.ones(n),                                   # home-court advantage
    strength_diff,                                # team quality
    dummies(margin_bucket),                       # score effects
    dummies(np.minimum(period[j], 5)),            # period (5 = overtime)
    poss["seconds_left"].to_numpy()[j] / 720,     # clock
    dummies(start_type) * sign[j][:, None],       # how the next possession starts, and whose it is
])

expected_extra = {}     # after[N] minus the expected value ("residual"), home team's view
for N in WINDOWS:
    rows = ~np.isnan(after[N])
    XtX = X_base[rows].T @ X_base[rows]
    beta = np.linalg.pinv(XtX) @ (X_base[rows].T @ after[N][rows])
    expected_extra[N] = after[N] - X_base @ beta

# ============================================================ find moments
# Unanswered runs. Walk through the scoring possessions of a game; a run
# continues while the same team keeps scoring and ends when the other scores.
scorer = np.sign(net)
s_idx = np.flatnonzero(scorer != 0)
new_run = np.r_[True, (scorer[s_idx][1:] != scorer[s_idx][:-1]) | (game[s_idx][1:] != game[s_idx][:-1])]
run_total = pd.Series(np.abs(net[s_idx])).groupby(np.cumsum(new_run)).cumsum().to_numpy()
run_pts = np.full(n, np.nan)        # unanswered points so far, at each scoring possession
run_pts[s_idx] = run_total
# carry the current run forward over scoreless possessions (needed for timeouts)
live = pd.DataFrame({"g": game, "pts": run_pts, "who": np.where(scorer != 0, scorer, np.nan)})
live_pts = live.groupby("g")["pts"].ffill().fillna(0).to_numpy()
live_who = live.groupby("g")["who"].ffill().fillna(0).to_numpy()

moments = []      # (label, counts toward the multiple-testing family?, possession index, team sign)
for K in RUN_SIZES:
    hit = s_idx[(run_total >= K) & (run_total - np.abs(net[s_idx]) < K)]   # the possession that made it K-0
    moments.append((f"{K}-0 run", True, hit, scorer[hit]))

# Highlight plays, from the play-by-play text linked to each possession.
row_of = pd.Series(idx, index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
linked = events[events["poss_num"].notna()].copy()
linked["i"] = row_of.reindex(pd.MultiIndex.from_frame(
    linked[["game_id", "poss_num"]].astype("int64"))).to_numpy()
linked = linked.dropna(subset=["i"]).astype({"i": int})
text = linked["home_txt"].fillna("") + " | " + linked["away_txt"].fillna("")


def plays(mask):
    return np.unique(linked.loc[mask, "i"].to_numpy())


made_shot = linked["etype"] == 1
i_dunk = plays(made_shot & text.str.contains("Dunk"))
i_block = plays((linked["etype"] == 2) & text.str.contains("BLOCK"))
i_steal = plays((linked["etype"] == 5) & text.str.contains("STEAL"))
i_ft1of1 = plays((linked["etype"] == 3) & text.str.contains("Free Throw 1 of 1"))
made_fg = (poss.get("fg2m", 0) + poss.get("fg3m", 0)).to_numpy() > 0
i_and1 = i_ft1of1[made_fg[i_ft1of1]]
# a steal "leading to a score": the stealing team scores on its next possession
nxt = i_steal + 1
nxt = nxt[nxt < n]
nxt = nxt[(game[nxt] == game[nxt - 1]) & (sign[nxt] == -sign[nxt - 1]) & (pts_off[nxt] > 0)]

moments += [
    ("dunk", True, i_dunk, sign[i_dunk]),                    # credit: the offense
    ("block", True, i_block, -sign[i_block]),                # credit: the defense
    ("steal -> score", True, nxt, sign[nxt]),
    ("and-one", True, i_and1, sign[i_and1]),
    # Not momentum claims - yardsticks. If a dunk "swings momentum", it should
    # beat an ordinary basket.
    ("(ordinary 2-pt basket)", False, np.setdiff1d(plays(made_shot & (linked["pts"] == 2)), i_dunk), None),
    ("(made 3-pointer)", False, plays(made_shot & (linked["pts"] == 3)), None),
]

# ====================================================== A. event study
gidx = pd.factorize(game)[0]
results = []
for label, in_family, i, team in moments:
    team = sign[i] if team is None else team
    for N in WINDOWS:
        x = expected_extra[N][i] * team             # flip to the moment-maker's point of view
        keep = ~np.isnan(x)
        est, se = ols_cluster(np.ones((keep.sum(), 1)), x[keep], gidx[i][keep])
        results.append(dict(moment=label, family=in_family, next_n=N, n=int(keep.sum()),
                            before=np.nanmean(before[N][i] * team), after=np.nanmean((after[N][i] * team)[keep]),
                            adjusted=est[0], se=se[0]))

# ====================================================== B. timeouts
# Put each timeout just before the first possession that starts at or after
# it (same game and period), and require that possession to be marked
# "Off Timeout" in the possession data.
to = events[events["etype"] == 9].copy()
to["side"] = np.where(to["home_txt"].notna(), 1, -1)        # who called it
game_rank = pd.Series(np.arange(len(np.unique(game))), index=np.unique(game))
clock_key = gidx * 10_000 + period * 1_000 + (720 - poss["seconds_left"].to_numpy())
to = to[to["game_id"].isin(game_rank.index)]
to_key = (game_rank.reindex(to["game_id"]).to_numpy() * 10_000 + to["period"].to_numpy() * 1_000
          + (720 - to["seconds_left"].to_numpy()))
tj = np.searchsorted(clock_key, to_key, side="left")
inside = tj < n
tj, to_side = tj[inside], to["side"].to_numpy()[inside]
same = (game[tj] == to["game_id"].to_numpy()[inside]) & (period[tj] == to["period"].to_numpy()[inside])
tj, to_side = tj[same], to_side[same]
caller = np.zeros(n)                       # +1 / -1 = who called a timeout right before possession j
caller[tj] = to_side
off_timeout = poss["starttype"].fillna("").to_numpy() == "Off Timeout"

victim = -live_who                         # the team being run on, after possession i
base = (live_pts >= TIMEOUT_RUN) & (idx + 1 < n)
called = base & off_timeout[j] & (caller[j] == victim)        # victim called timeout
no_timeout = base & ~off_timeout[j] & (caller[j] == 0)
for N in WINDOWS:
    rows = (called | no_timeout) & ~np.isnan(after[N])
    v = victim[rows]
    y = after[N][rows] * v                                    # victim's margin over the next N
    X = np.column_stack([
        np.ones(rows.sum()),
        called[rows],                                         # <- the timeout effect
        np.minimum(live_pts[rows], 14),                       # size of the run
        home_margin[rows] * v, strength_diff[rows] * v, v,    # score, team quality, home court
        sign[j][rows] * v,                                    # does the victim have the ball next?
        (scorer[rows] == live_who[rows]),                     # did the run team just score?
        dummies(np.minimum(period[j][rows], 5)),
        poss["seconds_left"].to_numpy()[j][rows] / 720,
    ]).astype(float)
    beta, se = ols_cluster(X, y, gidx[rows])
    results.append(dict(moment=f"timeout during {TIMEOUT_RUN}+ run", family=True, next_n=N,
                        n=int(called[rows].sum()), before=np.nan,
                        after=y[called[rows]].mean(), adjusted=beta[1], se=se[1],
                        no_timeout_after=y[~called[rows]].mean(), n_no_timeout=int((~called[rows]).sum())))

res = pd.DataFrame(results)
res["lo"], res["hi"] = res["adjusted"] - 1.96 * res["se"], res["adjusted"] + 1.96 * res["se"]
res["p"] = [erfc(abs(z) / sqrt(2)) for z in res["adjusted"] / res["se"]]
# Holm correction: sort the family's p-values; the smallest is multiplied by
# the number of tests, the next by one fewer, and so on.
fam = res[res["family"]].sort_values("p")
holm = np.minimum(np.maximum.accumulate(fam["p"].to_numpy() * (len(fam) - np.arange(len(fam)))), 1)
res["p_holm"] = pd.Series(holm, index=fam.index)

pd.set_option("display.width", 200)
show = res[["moment", "next_n", "n", "before", "after", "adjusted", "lo", "hi", "p", "p_holm"]]
print(f"\nPossessions: {n:,} in {len(game_rank):,} games.  Unit: points per 100 possessions.\n")
print("A/B. Effect of each moment on the next N possessions (for the team that made it;")
print("     for timeouts: for the team that called it, vs. no timeout in the same spot)\n")
print(show.to_string(index=False, na_rep="", formatters={
    **{c: "{:+.1f}".format for c in ("before", "after", "adjusted", "lo", "hi")},
    "n": "{:,}".format, "p": "{:.3f}".format, "p_holm": "{:.3f}".format}))
t = res[res["moment"].str.startswith("timeout")]
print("\nTimeout detail - raw margin of the team being run on, over the next N possessions:")
for _, r in t.iterrows():
    print(f"  next {r['next_n']:>2}: with timeout {r['after']:+.1f} (n={r['n']:,})   "
          f"without {r['no_timeout_after']:+.1f} (n={int(r['n_no_timeout']):,})")

# ====================================================== C. shuffle test
# Keep only games where every possession passed the data check.
clean = ~pd.Series(~ok).groupby(game).transform("any").to_numpy()
g_c, sign_c, per_c = gidx[clean], sign[clean], period[clean]
val = (pts_off - pts_def)[clean]                    # offense's net points on each possession
n_games = len(np.unique(g_c))
THRESHOLDS = (8, 10, 12)

team_game = g_c * 2 + (sign_c == 1)                 # one id per team per game
by_team = np.argsort(team_game, kind="stable")      # each team's possessions, in game order
pair = team_game[by_team][1:] == team_game[by_team][:-1]


def momentum_stats(v):
    """Runs per game at each threshold + 'does a team's possession predict its next one?'"""
    net_c = v * sign_c
    k = np.flatnonzero(net_c != 0)
    s = np.sign(net_c[k])
    starts = np.flatnonzero(np.r_[True, (s[1:] != s[:-1]) | (g_c[k][1:] != g_c[k][:-1])])
    totals = np.add.reduceat(np.abs(net_c[k]), starts)
    x = v[by_team]
    x = x - pd.Series(x).groupby(team_game[by_team]).transform("mean").to_numpy()
    lag1 = (x[1:] * x[:-1])[pair].sum() / (x ** 2).sum()
    return [(totals >= K).sum() / n_games for K in THRESHOLDS] + [lag1]


def shuffled(group):
    """Shuffle values among possessions that share the same group id."""
    order = np.argsort(group, kind="stable")
    g_sorted = group[order]
    out = np.empty_like(val)
    stats = []
    for _ in range(N_SHUFFLES):
        perm = np.argsort(g_sorted + rng.random(len(g_sorted)))   # random order inside each group
        out[order] = val[order][perm]
        stats.append(momentum_stats(out))
    return np.array(stats)


real = np.array(momentum_stats(val))
names = [f"{K}-0 runs per game" for K in THRESHOLDS] + ["link between a team's back-to-back possessions"]
shuffle_rows = []
for how, group in (("whole game", team_game), ("within each quarter", team_game * 10 + np.minimum(per_c, 9))):
    sims = shuffled(group)
    for k, name in enumerate(names):
        shuffle_rows.append(dict(statistic=name, shuffled=how, real=real[k], chance=sims[:, k].mean(),
                                 chance_sd=sims[:, k].std(ddof=1),
                                 p_more_than_chance=(1 + (sims[:, k] >= real[k]).sum()) / (1 + N_SHUFFLES)))
shuf = pd.DataFrame(shuffle_rows)
shuf["real_vs_chance"] = shuf["real"] / shuf["chance"]
print(f"\nC. Shuffle test ({N_SHUFFLES} shuffles, {n_games:,} games). 'chance' = a world with no momentum.")
print("   'whole game' shuffles a team's possessions across the game; 'within each quarter'")
print("   keeps each quarter's results in that quarter (so who was on the floor, fatigue")
print("   and garbage time can't pose as momentum).\n")
print(shuf.to_string(index=False, float_format="{:.4f}".format))

if DB == "data/nba.db":       # save findings (summaries only - no raw data)
    Path("results").mkdir(exist_ok=True)
    res.drop(columns="family").to_csv("results/momentum_effects.csv", index=False, float_format="%.4f")
    shuf.to_csv("results/momentum_shuffle.csv", index=False, float_format="%.5f")
    print("\nSaved results/momentum_effects.csv and results/momentum_shuffle.csv")
