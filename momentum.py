"""
NBA Momentum Study - Step 4: momentum as a RATE.

Run after expected.py. No hand-picked cut-offs (no "8-0 run", no "next 6
possessions"). Everything is built from one quantity:

  RATE  points per possession, compared with what was expected on that
        possession (see expected.py), in points per 100 possessions.

  MOMENTUM of a team = its recent scoring rate minus the opponent's recent
        scoring rate, recent possessions counting more: after each of a
        team's own possessions,
            rate <- keep x rate + (1 - keep) x (this possession's result).
        This "exponential smoothing" is not an arbitrary recipe: it is the
        best possible running estimate of a team's current form IF form is
        a level that drifts over time and each possession is a noisy
        reading of it (the steady-state Kalman filter of a local-level
        model). Each team's rate is taken over ITS OWN possessions, so
        unequal possession counts don't matter, and a team starts every
        game at zero, so a rate built on few possessions stays near neutral.

  HALF-LIFE = how fast older possessions fade (keep = 0.5 ^ (1/half-life)).
        The data cannot estimate it. After Part A the script prints, for
        each layer, the half-life whose momentum best predicts the next
        possession, with a 95% range from re-drawing whole games
        (bootstrap): on the raw rates it is the shortest one tried, i.e.
        the very last possession - that is the possession chain, not
        momentum - and once the chain is removed the best value jumps
        around from one group of seasons to the next. So Part A shows every
        half-life from 1 to 32, and Parts B and D and the saved table use a
        declared REFERENCE half-life of 2.8 of a team's own possessions
        (about a minute and a half of play), with the key numbers repeated
        at half-lives 1 and 8.

Parts:
 A. Carry-over. If a team has +10 of momentum, how much of it shows up on
    the next possession? Four layers:
      raw              the plain rates
      chain removed    also accounting for HOW each possession started
                       (after a steal, a miss, a made basket...)
      players          also accounting for the ten players on the floor
      same lineups     a second, model-free check of the same thing:
                       shuffles that only move results within a stretch
                       with no substitutions
    "Chance" = the same number after shuffling each team's possession
    results within a quarter (a permutation test). This also removes the
    built-in bias of looking at streaks inside short sequences.
 B. Whose momentum matters? The next possession against (1) the team's own
    recent offense and (2) the opponent's recent offense, separately.
 C. Streakiness without cut-offs. Is the scoring margin over stretches of
    2, 4, 8, 16, 32 possessions more spread out than independent possessions
    would give? Real games vs shuffled games vs a simulated "chain world"
    that keeps only the mechanical link.
 D. Plays. After a dunk, block, steal-and-score or and-one, the team's
    momentum going forward and its rate on each of its next five trips.

Saves table `poss_momentum` (rates, momentum before and after every
possession) for players.py and coaches.py.

Usage:
    python momentum.py [--db data/fake_nba.db] [--shuffles 40] [--game 22400001]
    python momentum.py --half-life 4        # a different reference half-life (default 2.8)
"""

import sqlite3
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))    # so "python -I momentum.py" still finds rates.py next to it
from rates import Rates  # noqa: E402


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
B = arg("--shuffles", 40)
GAME = arg("--game", 0)
H = arg("--half-life", 2 ** 1.5)      # reference half-life for Parts B and D and the saved table (see docstring)
H_LO, H_HI = 1.0, 8.0                 # ... repeated at these two, to show how much the answers depend on it
HALF_LIVES = (1, 2, 4, 8, 16, 32)     # Part A: a full range, not a choice
EXPLORE = (2020, 2021, 2022)          # seasons for the half-life diagnostic; the other groups are checks
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)

# ============================================================== load data
con = sqlite3.connect(DB)
# make the database use the real (game, possession) indexes when joining tables; left to itself it
# sometimes builds a makeshift index on half the key, which makes loading take many minutes
con.execute("PRAGMA automatic_index = OFF")
try:
    poss = pd.read_sql_query(
        """SELECT p.*, e.exp_base, e.exp_chain, e.exp_lineup_chain
           FROM poss p JOIN poss_expected e USING (game_id, poss_num) ORDER BY p.game_id, p.poss_num""", con)
except Exception as e:
    raise SystemExit(f"Run build_lineups.py and expected.py first. [{e}]")
ev = pd.read_sql_query(
    """SELECT game_id, poss_num, period, seconds_left, eventmsgtype AS etype, pts,
              homedescription AS home_txt, visitordescription AS away_txt
       FROM events WHERE (poss_num IS NOT NULL AND eventmsgtype IN (1, 2, 3, 5)) OR eventmsgtype = 8""", con)
con.close()
if "playoffs" not in poss:
    poss["playoffs"] = 0

# Every possession that passed the data check is used. Blowouts ("garbage time") are NOT thrown
# out: dropping possessions because of the score selects on what happened before and biases every
# momentum measure (checked on simulated games). The expected-points model handles big late leads.
n = len(poss)
pos = np.arange(n)
game = poss["game_id"].to_numpy()
gidx = pd.factorize(game)[0]
G = gidx.max() + 1
period = poss["period"].to_numpy()
clock = poss["seconds_left"].to_numpy()
season = poss["season"].to_numpy()
playoffs = (poss["playoffs"] == 1).to_numpy()
home = (poss["is_home"] == 1).to_numpy()
sign = np.where(home, 1, -1)
points = poss["points"].to_numpy(float)
net = (poss["points"] - poss["def_points"]).to_numpy().astype(int)       # offense's net points
new_game = np.r_[True, gidx[1:] != gidx[:-1]]
first = np.r_[True, (gidx[1:] != gidx[:-1]) | (period[1:] != period[:-1])]   # first possession of a period
last = np.r_[first[1:], True]

SPLITS = {"2020-22": np.isin(season, EXPLORE) & ~playoffs, "2023-24": (season > max(EXPLORE)) & ~playoffs,
          "2015-19": (season < min(EXPLORE)) & ~playoffs, "playoffs": playoffs}
SPLITS = {k: v for k, v in SPLITS.items() if v.any()}
if len(SPLITS) == 1:
    SPLITS = {"all": np.ones(n, bool)}
choose = next(iter(SPLITS.values()))

# rate vs expectation on every possession, points per 100 possessions (see expected.py)
res_raw = (points - poss["exp_base"].to_numpy()) * 100                 # vs team quality and situation
res_adj = (points - poss["exp_chain"].to_numpy()) * 100                # ... chain removed
has_lineup = poss["exp_lineup_chain"].notna().to_numpy()
res_lin = np.where(has_lineup, (points - poss["exp_lineup_chain"].to_numpy()) * 100, 0.0)   # ... and the ten players
everything = np.ones(n, bool)
team_game = gidx * 2 + home

# ------------------------------------------------- momentum = recency-weighted rates
# The update itself lives in rates.py (see there for why it is a short loop and not one big formula).
past, future = Rates(gidx, home), Rates(gidx, home, backwards_in_time=True)


def momentum_before(x, half_life):
    """Before each possession: (recent offense rate of the team with the ball,
    recent offense rate of the team defending)."""
    h, a = past(x, half_life)
    h, a = (np.where(new_game, 0.0, np.r_[0.0, v[:-1]]) for v in (h, a))
    return np.where(home, h, a), np.where(home, a, h)


def momentum_ahead(x, half_life):
    """From each possession onward: (offense rate of the team with the ball,
    offense rate of the team defending) - the same weighting, looking forward."""
    h, a = future(x, half_life)
    return np.where(home, h, a), np.where(home, a, h)


# ------------------------------------------------------------ shuffle groups
game_no = pd.Series(np.arange(G), index=pd.unique(game))
key = gidx * 10_000 + period * 1_000 + (720 - clock)
subs = ev[(ev["etype"] == 8) & ev["game_id"].isin(game_no.index)]
sub_keys = np.unique(game_no.reindex(subs["game_id"]).to_numpy() * 10_000 + subs["period"].to_numpy() * 1_000
                     + (720 - subs["seconds_left"].to_numpy()))
stint_no = np.searchsorted(sub_keys, key, side="right")
new_stint = first | np.r_[True, stint_no[1:] != stint_no[:-1]]       # stint = stretch with no substitution
GROUPS = {"quarter": np.cumsum(first) * 2 + home, "stint": np.cumsum(new_stint) * 2 + home}


def shuffler(group):
    order = np.argsort(group, kind="stable")
    sorted_group = group[order].astype(float)

    def draw():
        new = np.empty(n, int)
        new[order] = order[np.argsort(sorted_group + rng.random(n))]
        return new
    return draw


DRAW = {name: shuffler(g) for name, g in GROUPS.items()}


def slope(x, y, mask):
    """Least-squares slope of y on x (how much of x shows up in y)."""
    x, y = x[mask], y[mask]
    x = x - x.mean()
    return (x * (y - y.mean())).sum() / (x * x).sum()


def carry_over(res, ok):
    """For every half-life and every group of games: slope of this possession's
    result on the offense's momentum (its recent rate minus the defense's)."""
    out = []
    for h in HALF_LIVES:
        off, de = momentum_before(res, h)
        out += [slope(off - de, res, m & ok) for m in SPLITS.values()]
    return np.array(out)


def versus_chance(stat, res, group, ok):
    real = stat(res, ok)
    sims = []
    for _ in range(B):
        order = DRAW[group]()
        sims.append(stat(res[order], ok[order]))
    sims = np.array(sims)
    return real - sims.mean(0), (real - sims.mean(0)) / sims.std(0, ddof=1)


print(f"{n:,} possessions in {G:,} games (only possessions that failed the data check are left out).")
print("Unit: points per 100 possessions.\n")

# ====================================================== A. carry-over
print("A. CARRY-OVER - share of a team's momentum that shows up on its next possession, minus chance.")
print("   (+0.02 = of a +10 momentum edge, +0.2 is still there on the next possession.)\n")
LAYERS = (("raw", res_raw, "quarter", everything), ("chain removed", res_adj, "quarter", everything),
          ("chain + players on the floor removed", res_lin, "quarter", has_lineup),
          ("chain removed, shuffled within same lineups", res_adj, "stint", everything))
results_a, z_choose = [], {}
for label, res, group, ok in LAYERS:
    excess, z = versus_chance(carry_over, res, group, ok)
    excess, z = excess.reshape(len(HALF_LIVES), -1), z.reshape(len(HALF_LIVES), -1)
    for s, split in enumerate(SPLITS):
        results_a.append({"layer": label, "seasons": split,
                          **{f"half-life {h}": f"{excess[i, s]:+.3f} (z {z[i, s]:+.1f})" for i, h in enumerate(HALF_LIVES)}})
    z_choose[label] = z[:, 0]
results_a = pd.DataFrame(results_a)
print(results_a.to_string(index=False))

# Can the data tell us the half-life? For every half-life on a fine grid: how well does its momentum predict
# the next possession (the correlation)? The best one in 2020-22, with a 95% range from re-drawing whole games
# with replacement (bootstrap), and the best one in each other group of seasons as a check. Shown for the raw
# rates and for the two adjusted versions. (The analyses below do NOT use the winner - see the docstring.)
fine = np.exp(np.linspace(np.log(0.5), np.log(64), 29))


def per_game_sums(res, ok):
    """For each group of seasons: (half-life, game, [sum of momentum x result, sum of momentum^2])."""
    sums = {name: np.zeros((len(fine), G, 2)) for name in SPLITS}
    for a, h in enumerate(fine):
        off, de = momentum_before(res, h)
        mom = off - de
        for name, mask in SPLITS.items():
            m_ = mask & ok
            sums[name][a, :, 0] = np.bincount(gidx[m_], weights=mom[m_] * res[m_], minlength=G)
            sums[name][a, :, 1] = np.bincount(gidx[m_], weights=mom[m_] ** 2, minlength=G)
    return {name: sums[name][:, np.unique(gidx[mask & ok])] for name, mask in SPLITS.items()}


def best(per_game, weights=1.0):
    xy, xx = (per_game[:, :, 0] * weights).sum(1), (per_game[:, :, 1] * weights).sum(1)
    return fine[np.argmax(xy / np.sqrt(xx))]


boot_rng = np.random.default_rng(1)          # own generator for the adjusted layers (keeps Parts B and C as before)
diagnostic = []
for k, (label, res, group, ok) in enumerate(LAYERS[:3]):
    pg = per_game_sums(res, ok)
    first_split = next(iter(SPLITS))
    n_g = pg[first_split].shape[1]
    draw = rng if k == 0 else boot_rng
    boot = np.array([best(pg[first_split], np.bincount(draw.integers(0, n_g, n_g), minlength=n_g)) for _ in range(500)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    row = {"layer": label, f"best half-life, {first_split} (95% range)": f"{best(pg[first_split]):.1f} ({lo:.1f} to {hi:.1f})"}
    row.update({name: f"{best(pg[name]):.1f}" for name in list(SPLITS)[1:]})
    diagnostic.append(row)
diagnostic = pd.DataFrame(diagnostic)
print("\n   Can the data pin down the half-life?  The one whose momentum best predicts the next possession (grid 0.5 to 64):")
print("   " + diagnostic.to_string(index=False).replace("\n", "\n   "))
print("   On the raw rates the winner is the last possession: that is the possession chain, not momentum. With the chain")
print("   removed the winner moves around from one group of seasons to the next, so the data do not pin it down.")
print(f"   Used below: a reference half-life of {H:.1f} of a team's own trips; key numbers repeated at {H_LO:g} and {H_HI:g}.")

# ====================================================== B. whose momentum matters?
def two_slopes(res, ok):
    """This possession's result against (own recent offense, opponent's recent offense) together."""
    off, de = momentum_before(res, H)
    out = []
    for m in SPLITS.values():
        m = m & ok
        X = np.column_stack([np.ones(m.sum()), off[m], de[m]])
        out += list(np.linalg.lstsq(X, res[m], rcond=None)[0][1:])
    return np.array(out)


print("\n\nB. WHOSE MOMENTUM MATTERS?  The team with the ball: its next possession against ...")
print("   'own' = its own recent scoring (a hot offense staying hot);")
print("   'opponent' = the other team's recent scoring (their offense feeding their defense).  Minus chance.\n")
results_b = []
for label, res, group, ok in LAYERS:
    excess, z = versus_chance(two_slopes, res, group, ok)
    for s, split in enumerate(SPLITS):
        results_b.append({"layer": label, "seasons": split,
                          "own recent offense": f"{excess[2 * s]:+.3f} (z {z[2 * s]:+.1f})",
                          "opponent's recent offense": f"{excess[2 * s + 1]:+.3f} (z {z[2 * s + 1]:+.1f})"})
results_b = pd.DataFrame(results_b)
print(results_b.to_string(index=False))

# ====================================================== C. streakiness without cut-offs
STRETCHES = (2, 4, 8, 16, 32)
period_id = np.cumsum(first)
slot_in_period = pos - np.maximum.accumulate(np.where(first, pos, 0))


def spread_ratio(v_home, mask):
    """Variance of the home margin over stretches of k possessions, divided by k x the
    variance of one possession. 1 = independent possessions; above 1 = streaky."""
    one = v_home[mask].var()
    out = []
    for k in STRETCHES:
        block = period_id * 1000 + slot_in_period // k
        full = pd.Series(mask.astype(int)).groupby(block).transform("sum").to_numpy() == k   # complete stretches only
        sums = pd.Series(v_home[full]).groupby(block[full]).sum()
        out.append(sums.var() / (k * one))
    return np.array(out)


# simulated worlds: replay each game slot by slot, copying each possession's result from a random
# possession by the same team in the same game - in the chain world, one that also started the same way
st = poss["starttype"].fillna("Off Dead Ball")
cls = np.select([st == "Off Steal", st.str.contains("Block"), st.str.contains("Miss") & ~st.str.contains("FT"),
                 st.str.contains("Make") & ~st.str.contains("FT"), st.str.contains("FT"), st == "Off Timeout"],
                [0, 1, 2, 3, 4, 6], 5)
NC = 7
next_cls = np.r_[cls[1:], 0]
slot = pos - np.maximum.accumulate(np.where(new_game, pos, 0))
P = slot.max() + 1


def donor_pool(pool_key, n_pools):
    d = pd.DataFrame({"k": pool_key, "net": net, "next": next_cls})[~last]
    d["j"] = d.groupby("k").cumcount()
    size = np.zeros(n_pools, int)
    counts = d.groupby("k").size()
    size[counts.index] = counts.to_numpy()
    pts, nxt = (np.zeros((n_pools, max(size.max(), 1)), np.int8) for _ in range(2))
    pts[d["k"], d["j"]], nxt[d["k"], d["j"]] = d["net"], d["next"]
    return size, pts, nxt


size_c, pts_c, nxt_c = donor_pool(team_game * NC + cls, G * 2 * NC)
size_a, pts_a, nxt_a = donor_pool(team_game, G * 2)
grid = {name: np.zeros((G, P), dtype) for name, dtype in
        (("team", int), ("first", bool), ("last", bool), ("cls", int), ("net", int))}
for name, v in (("team", team_game), ("first", first), ("last", last), ("cls", cls), ("net", net)):
    grid[name][gidx, slot] = v


def replay(keep_chain):
    sim = np.zeros((G, P), int)
    cur = grid["cls"][:, 0].copy()
    for p in range(P):
        cur = np.where(grid["first"][:, p], grid["cls"][:, p], cur)
        kc, ka = grid["team"][:, p] * NC + cur, grid["team"][:, p]
        u = rng.random(G)
        has = (size_c[kc] > 0) & keep_chain
        jc, ja = (u * size_c[kc]).astype(int), (u * np.maximum(size_a[ka], 1)).astype(int)
        pts = np.where(has, pts_c[kc, jc], pts_a[ka, ja])
        cur = np.where(has, nxt_c[kc, jc], nxt_a[ka, ja])
        sim[:, p] = np.where(grid["last"][:, p], grid["net"][:, p], pts)
    return sim[gidx, slot]


print("\n\nC. STREAKINESS WITHOUT CUT-OFFS - how spread out is the scoring margin over stretches of k possessions,")
print("   compared with independent possessions?  'excess' = real games vs shuffled games; 'the chain adds' = a")
print("   simulated world with only the mechanical link vs one without it; 'left over' = what the chain doesn't explain.\n")
worlds = {"shuffled": [net[DRAW["quarter"]()] * sign for _ in range(B)],
          "no link": [replay(False) * sign for _ in range(B)], "chain": [replay(True) * sign for _ in range(B)]}
results_c = []
for split, mask in SPLITS.items():
    real = spread_ratio(net * sign, mask)
    sim = {w: np.array([spread_ratio(v, mask) for v in vs]) for w, vs in worlds.items()}
    excess, adds = real / sim["shuffled"].mean(0), sim["chain"].mean(0) / sim["no link"].mean(0)
    noise = sim["shuffled"].std(0, ddof=1) / sim["shuffled"].mean(0)
    for i, k in enumerate(STRETCHES):
        results_c.append({"seasons": split, "stretch": f"{k} possessions",
                          "excess over chance": f"{excess[i] - 1:+.1%} (z {(excess[i] - 1) / noise[i]:+.1f})",
                          "the chain adds": f"{adds[i] - 1:+.1%}",
                          "left over": f"{excess[i] / adds[i] - 1:+.1%} (z {(excess[i] / adds[i] - 1) / noise[i]:+.1f})"})
results_c = pd.DataFrame(results_c)
print(results_c.to_string(index=False))

# ====================================================== D. plays
# momentum ahead of every possession, from the point of view of the team with the ball
own_raw, opp_raw = momentum_ahead(res_raw, H)
own_adj, opp_adj = momentum_ahead(res_adj, H)
ahead_raw, ahead_adj = own_raw - opp_raw, own_adj - opp_adj
ahead_lin = np.subtract(*momentum_ahead(res_lin, H))
ahead_lin_lo, ahead_lin_hi = (np.subtract(*momentum_ahead(res_lin, h)) for h in (H_LO, H_HI))   # sensitivity
m_off, m_def = momentum_before(res_raw, H)

row_of = pd.Series(pos, index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
linked = ev[ev["poss_num"].notna()].copy()
linked["i"] = row_of.reindex(pd.MultiIndex.from_frame(linked[["game_id", "poss_num"]].astype("int64"))).to_numpy()
linked = linked.dropna(subset=["i"]).astype({"i": int})
text = linked["home_txt"].fillna("") + " | " + linked["away_txt"].fillna("")


def plays(mask):
    return np.unique(linked.loc[mask, "i"].to_numpy())


made_shot = linked["etype"] == 1
i_dunk = plays(made_shot & text.str.contains("Dunk"))
i_block = plays((linked["etype"] == 2) & text.str.contains("BLOCK"))
i_steal = plays((linked["etype"] == 5) & text.str.contains("STEAL"))
i_ft = plays((linked["etype"] == 3) & text.str.contains("Free Throw 1 of 1"))
i_and1 = i_ft[((poss["fg2m"] + poss["fg3m"]).to_numpy() > 0)[i_ft]]
nxt = i_steal + 1
nxt = nxt[nxt < n]
nxt = nxt[(gidx[nxt] == gidx[nxt - 1]) & (sign[nxt] == -sign[nxt - 1]) & (points[nxt] > 0)]   # stealing team scores next
MOMENTS = [("dunk", True, i_dunk, sign[i_dunk]), ("block", True, i_block, -sign[i_block]),
           ("steal -> score", True, nxt, sign[nxt]), ("and-one", True, i_and1, sign[i_and1]),
           ("(ordinary 2-pt basket)", False, np.setdiff1d(plays(made_shot & (linked["pts"] == 2)), i_dunk), None),
           ("(made 3-pointer)", False, plays(made_shot & (linked["pts"] == 3)), None)]

# the team's k-th trip after a play = its k-th offensive possession and the opponent's k-th
rank = np.empty(n, int)                                   # position of each possession in its team's own sequence
rank[past.order] = pos
seq_key = team_game[past.order] * 1_000_000 + slot[past.order]          # sorted (team-game, time)
res_adj_seq = res_lin[past.order]
tg_seq = team_game[past.order]


def trip_rates(i, team, trips=5):
    """Chain-removed net rate on each of the team's next trips after possession i."""
    out = []
    side = (team == 1).astype(int)
    own_tg, opp_tg = gidx[i] * 2 + side, gidx[i] * 2 + (1 - side)
    own_at = np.searchsorted(seq_key, own_tg * 1_000_000 + slot[i] + 1)      # team's first offensive possession after i
    opp_at = np.searchsorted(seq_key, opp_tg * 1_000_000 + slot[i] + 1)
    for k in range(trips):
        a, b = np.minimum(own_at + k, n - 1), np.minimum(opp_at + k, n - 1)
        ok = (tg_seq[a] == own_tg) & (tg_seq[b] == opp_tg) & (own_at + k < n) & (opp_at + k < n)
        out.append(np.where(ok, res_adj_seq[a] - res_adj_seq[b], np.nan))
    return out


def mean_ci(x, groups):
    keep = ~np.isnan(x)
    x, groups = x[keep], groups[keep]
    per = pd.DataFrame({"g": groups, "x": x}).groupby("g")["x"].agg(["sum", "size"])
    mean = x.mean()
    se = np.sqrt(((per["sum"] - mean * per["size"]) ** 2).sum()) / per["size"].sum()
    return mean, se


print(f"\n\nD. PLAYS - what the team's rate does afterwards (it = the team that made the play).")
print(f"   'momentum ahead' = its recency-weighted net rate from the next possession on (half-life {H:.1f}).\n")
results_d = []
for label, in_family, i, team in MOMENTS:
    team = sign[i] if team is None else team
    j = i + 1
    keep = (j < n) & (gidx[np.minimum(j, n - 1)] == gidx[i])
    i, team, j = i[keep], team[keep], j[keep]
    flip = team * sign[j]                                     # +1 if the team has the ball on the next possession
    row = {"play": label, "family": in_family, "n": len(i),
           "momentum before": np.mean((m_off[i] - m_def[i]) * team * sign[i])}
    for name, v in (("momentum ahead, raw", ahead_raw), ("chain removed", ahead_adj), ("chain + players removed", ahead_lin)):
        mean, se = mean_ci(v[j] * flip, gidx[i])
        row[name], row[name + " se"] = mean, se
    for name, v in ((f"same at half-life {H_LO:g}", ahead_lin_lo), (f"same at half-life {H_HI:g}", ahead_lin_hi)):
        row[name] = mean_ci(v[j] * flip, gidx[i])[0]
    for k, r in enumerate(trip_rates(i, team), 1):
        row[f"trip {k}"] = np.nanmean(r)
    results_d.append(row)
res_d = pd.DataFrame(results_d)
col = "chain + players removed"
res_d["lo"], res_d["hi"] = res_d[col] - 1.96 * res_d[col + " se"], res_d[col] + 1.96 * res_d[col + " se"]
res_d["p"] = [erfc(abs(z) / sqrt(2)) for z in res_d[col] / res_d[col + " se"]]
fam = res_d[res_d["family"]].sort_values("p")               # Holm correction across the highlight plays
res_d["p_holm"] = pd.Series(np.minimum(np.maximum.accumulate(fam["p"].to_numpy() * (len(fam) - np.arange(len(fam)))), 1), index=fam.index)
show = res_d[["play", "n", "momentum before", "momentum ahead, raw", "chain removed", col, "lo", "hi", "p_holm",
              f"same at half-life {H_LO:g}", f"same at half-life {H_HI:g}"] + [f"trip {k}" for k in range(1, 6)]]
print(show.to_string(index=False, na_rep="", formatters={
    "n": "{:,}".format, "p_holm": "{:.3f}".format,
    **{c: "{:+.1f}".format for c in show.columns if c not in ("play", "n", "p_holm")}}))
print("   ('chain + players removed' also accounts for who was on the floor; lo/hi = its 95% range; 'same at half-life'")
print("    = that column with a shorter / longer memory; trips 1-5 = that same rate on each of the team's next five trips)")

# ====================================================== save for the other scripts
con = sqlite3.connect(DB)
pd.DataFrame({"game_id": game, "poss_num": poss["poss_num"],
              "rate_raw": res_raw, "rate_adj": res_adj, "rate_lineup": np.where(has_lineup, res_lin, np.nan),
              "before_off": m_off, "before_def": m_def,                 # recent offense rate: team with ball / defending
              "ahead_own": own_raw, "ahead_opp": opp_raw,               # same, looking forward (raw)
              "ahead_adj": ahead_adj, "ahead_lineup": ahead_lin,        # net, chain removed / chain + players removed
              # ... and the same at a shorter and a longer half-life (H_LO, H_HI), as a robustness check
              "ahead_lineup_lo": ahead_lin_lo, "ahead_lineup_hi": ahead_lin_hi}
             ).to_sql("poss_momentum", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_pm ON poss_momentum(game_id, poss_num)")
pd.DataFrame({"name": ["half_life", "half_life_lo", "half_life_hi"], "value": [H, H_LO, H_HI]}
             ).to_sql("momentum_settings", con, if_exists="replace", index=False)
con.commit()
con.close()
print("\nSaved table poss_momentum. Next: python players.py / python coaches.py")

# ====================================================== one game's momentum
if GAME:
    g = np.flatnonzero(game == GAME)
    home_mom = (m_off - m_def)[g] * sign[g]                    # home team's momentum before each possession
    teams = poss.loc[g[0], ["home", "away"]]
    print(f"\n\nMOMENTUM in game {GAME}: {teams['away']} at {teams['home']}  (half-life {H:.1f})")
    print("  Biggest swings - the five moments when one team's momentum peaked:")
    order, picked = np.argsort(-np.abs(home_mom)), []
    for k in order:                                           # peaks at least 10 possessions apart
        if all(abs(k - q) >= 10 for q in picked):
            picked.append(k)
        if len(picked) == 5:
            break
    for k in sorted(picked):
        i = g[k]
        who = teams["home"] if home_mom[k] > 0 else teams["away"]
        print(f"    Q{period[i]} {int(clock[i]) // 60}:{int(clock[i]) % 60:02d}  {who} {abs(home_mom[k]):+.0f} per 100 "
              f"(score margin for {teams['home']}: {int(poss.loc[i, 'start_margin'] * sign[i]):+d})")

if DB == "data/nba.db":
    Path("results").mkdir(exist_ok=True)
    for old in ("momentum_effects.csv", "momentum_shuffle.csv", "chain_correlation_ladder.csv", "chain_during_run.csv"):
        Path("results", old).unlink(missing_ok=True)
    results_a.to_csv("results/momentum_carry_over.csv", index=False)
    results_b.to_csv("results/momentum_whose.csv", index=False)
    results_c.to_csv("results/momentum_streakiness.csv", index=False)
    res_d.drop(columns="family").to_csv("results/momentum_plays.csv", index=False, float_format="%.3f")
    print("Saved results/momentum_*.csv")
