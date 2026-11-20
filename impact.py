"""
NBA Momentum Study - Step 6b: OVERALL IMPACT, measured as precisely as the data allow.

OVERALL IMPACT (printed by players.py) is a player's effect on his team's scoring margin per 100
possessions while he is on the floor: adjusted plus-minus, a measure of quality, not of momentum.
The first version averaged one ridge estimate per season (expected.py's player_effects), and its
rank ranges were wide (Curry anywhere from 1st to 28th). This script uses two standard ways to get
more out of the same data, and keeps each one only as far as it makes the player estimates predict
games they have not seen better.

 1. All of a player's seasons together (multi-level ridge regression).
    Before, each season was pulled toward zero on its own, as if a player's level one season said
    nothing about the next. It does: players repeat. Pulling each season toward zero separately and
    then averaging over-shrinks a long career and claims more certainty than there is, because ten
    separate pulls toward "average" count as ten pieces of evidence that he is average. Here a
    player's effect in a season = his CAREER level + that SEASON's deviation from it, and each part
    has its own amount of shrinkage. A career level is estimated from all his seasons at once, and
    roster changes between seasons help separate teammates who usually play together.
    (Ridge regression is the posterior mean when every effect has a normal prior centred on zero;
    the shrinkage amount is sigma^2 / tau^2, noise variance over prior variance. Two parts = two
    prior variances, one for career levels and one for season deviations.)

 2. Shot-making luck taken out (luck adjustment).
    Whether threes and free throws go in is about 43% of the variance of points per possession. A
    defender has no control over an opponent's free throws and little over whether a three drops,
    so most of that only blurs the estimates. Each three-point attempt and free throw can be valued
    at the shooter's expected percentage instead of the make or miss: his season rate pulled toward
    the league rate (beta-binomial model; how hard to pull comes from the spread of the players'
    rates, method of moments). The shooter keeps the credit for his own shooting skill; only the
    luck goes. It also takes out some real signal (a player who creates open threes for teammates),
    so the SHARE of the luck that is taken out (0 = none ... 1 = all) is chosen by the same test as
    everything else.

The test. Every setting (the two shrinkage amounts and the share of luck taken out) is judged by
how well the player effects predict the ACTUAL points of possessions in games left out of the fit
(5-fold cross-validation by game, the same folds as expected.py). Narrower intervals only count if
the estimates also predict better. The old method is one of the settings (no career part, no luck
taken out, each season's shrinkage chosen as in expected.py), so the comparison is direct, and this
script checks that it reproduces expected.py's numbers.

Uncertainty: each player's "+/-" comes from the posterior distribution of the model, with the career
and season parts and the link between his seasons taken into account. The number shown is the average
of his seasons, each counting by how many possessions he played in it (a rate, not a total), measured
against the average player of the same season (positive = better than average). In simulated leagues
with known truth (tests/test_impact.py) these 95% intervals held the true value for 87-92% of players,
so they are slightly optimistic; the old way's held it for only 78-82%.

Run after expected.py.   Usage:  python impact.py [--db data/nba.db]
    Sensitivity check (no cross-validation, nothing saved):  --lam-season 10000 --lam-career 3162 --luck 0.25
Saves tables player_impact (one row per player) and player_season_impact (one row per season played).
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))      # so the shared module is found under python -I
from situation import situation_matrix  # noqa: E402


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
FOLDS = 5
LAM_SEASON = np.logspace(0, 5, 21)                  # shrinkage of a season's effect (expected.py's grid)
LAM_CAREER = np.r_[np.logspace(0, 5, 11), np.inf]    # shrinkage of the career level; inf = no career level (the old way)
LUCK = np.linspace(0, 1, 5)                         # share of shot-making luck taken out
pd.set_option("display.width", 250)

# ============================================================== load data
con = sqlite3.connect(DB)
con.execute("PRAGMA automatic_index = OFF")
try:
    poss = pd.read_sql_query(
        """SELECT p.*, l.o1, l.o2, l.o3, l.o4, l.o5, l.d1, l.d2, l.d3, l.d4, l.d5, l.complete
           FROM poss p LEFT JOIN lineups l USING (game_id, poss_num)
           WHERE p.points_ok = 1 ORDER BY p.game_id, p.poss_num""", con)
    old_effects = pd.read_sql_query("SELECT * FROM player_effects", con)
    names = pd.read_sql_query("SELECT * FROM players", con).drop_duplicates("player_id").set_index("player_id")["player"]
except Exception as e:
    raise SystemExit(f"Run build_lineups.py and expected.py first. [{e}]")
if "playoffs" not in poss:
    poss["playoffs"] = 0
situation, _, _ = situation_matrix(con, poss)                  # exactly the columns expected.py uses
shots = pd.read_sql_query(
    """SELECT game_id, poss_num, eventmsgtype AS etype, player1_id AS shooter, player1_team_abbreviation AS team,
              COALESCE(homedescription, '') || ' ' || COALESCE(visitordescription, '') AS txt
       FROM events WHERE poss_num IS NOT NULL AND eventmsgtype IN (1, 2, 3)""", con)
con.close()

use = ((poss["complete"] == 1) & (poss["playoffs"] == 0)).to_numpy()     # regular season, all ten players known
poss, C = poss[use].reset_index(drop=True), situation[use]
n = len(poss)
y = poss["points"].to_numpy(float)
season = poss["season"].to_numpy()
seasons = np.unique(season)
off = poss[[f"o{k}" for k in range(1, 6)]].to_numpy().astype("int64")
dfn = poss[[f"d{k}" for k in range(1, 6)]].to_numpy().astype("int64")
game_code, game_ids = pd.factorize(poss["game_id"])
fold = np.zeros(n, int)
for s in seasons:                                              # the same folds as expected.py
    r = season == s
    fold[r] = pd.factorize(poss["game_id"].to_numpy()[r])[0] % FOLDS
print(f"{n:,} regular-season possessions with all ten players known, {len(seasons)} season(s), {len(game_ids):,} games")

# ============================================================== shot-making luck
row_of = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
shots["row"] = row_of.reindex(pd.MultiIndex.from_frame(shots[["game_id", "poss_num"]].astype("int64"))).to_numpy()
shots = shots.dropna(subset=["row"]).astype({"row": int})
shots = shots[shots["team"].to_numpy() == poss["offense"].to_numpy()[shots["row"]]]     # the team with the ball
is3 = (shots["etype"].isin([1, 2]) & shots["txt"].str.contains("3PT", regex=False)).to_numpy()
isft = (shots["etype"] == 3).to_numpy()
shots = shots[is3 | isft].copy()
shots["kind"] = np.where(is3[is3 | isft], "3PT", "FT")
shots["value"] = np.where(shots["kind"] == "3PT", 3.0, 1.0)
shots["made"] = np.where(shots["kind"] == "3PT", shots["etype"] == 1,
                         ~shots["txt"].str.contains("MISS", regex=False)).astype(float)
shots["season"] = season[shots["row"]]
shots["fold"] = fold[shots["row"]]


def luck_points(leave_out=None, report=False):
    """For every possession: expected minus actual points of its threes and free throws, each valued at the
    shooter's expected percentage = his season rate pulled toward the league rate (beta-binomial). The pull
    (a prior worth k attempts) is estimated per season from how much more players' rates differ than luck
    allows (method of moments). Percentages come only from games outside fold `leave_out`."""
    pool = shots if leave_out is None else shots[shots["fold"] != leave_out]
    pct = []
    for (kind, s), d in pool.groupby(["kind", "season"]):
        g = d.groupby("shooter")["made"].agg(["sum", "count"])
        p = g["sum"].sum() / g["count"].sum()
        real_var = ((g["count"] * (g["sum"] / g["count"] - p) ** 2).sum() - len(g) * p * (1 - p)) / g["count"].sum()
        k = p * (1 - p) / real_var - 1 if real_var > 0 else np.inf
        est = (g["sum"] + k * p) / (g["count"] + k) if np.isfinite(k) else pd.Series(p, index=g.index)
        pct.append(pd.DataFrame({"kind": kind, "season": s, "shooter": g.index, "p": est.to_numpy()}))
        if report:
            print(f"  {s} {kind}: league {p:.3f}, real spread between shooters {np.sqrt(max(real_var, 0)):.3f}, "
                  f"a shooter's rate is pulled toward the league rate as if he had {k:,.0f} more attempts at it")
    pct = pd.concat(pct).set_index(["kind", "season", "shooter"])["p"]
    p_shot = pct.reindex(pd.MultiIndex.from_frame(pool[["kind", "season", "shooter"]])).to_numpy()
    return np.bincount(pool["row"], weights=pool["value"].to_numpy() * (p_shot - pool["made"].to_numpy()), minlength=n)


print("\nShot-making luck: threes and free throws valued at the shooter's expected percentage")
luck_all = luck_points(report=len(seasons) <= 3)
print(f"  points per possession: variance {y.var():.3f}; with all of that luck taken out {(y + luck_all).var():.3f} "
      f"({1 - (y + luck_all).var() / y.var():.0%} of the variance is shot-making luck)")
luck_fold = [luck_points(f) for f in range(FOLDS)]


# ============================================================== regression pieces, season by season
def gram(cols, Cs, m):
    """X'X for rows whose player columns are cols (rows x 10) and situation columns Cs, without building X."""
    q = Cs.shape[1]
    A = np.zeros((m + q, m + q))
    A[:m, :m] = np.bincount((cols[:, :, None] * m + cols[:, None, :]).ravel(), minlength=m * m).reshape(m, m)
    flat = cols.ravel()
    for c in range(q):
        A[:m, m + c] = np.bincount(flat, weights=np.repeat(Cs[:, c], cols.shape[1]), minlength=m)
    A[m:, :m] = A[:m, m:].T
    A[m:, m:] = Cs.T @ Cs
    return A


def cross(cols, Cs, v, m):
    """X'v."""
    return np.r_[np.bincount(cols.ravel(), weights=np.repeat(v, cols.shape[1]), minlength=m), Cs.T @ v]


def independent(Cs):
    """Situation columns that are not exact combinations of other ones. Some are: "4th quarter or overtime" =
    "period 4" + "period 5". That changes no prediction, but it makes the equations nearly singular, and this
    script inverts them, so such columns (and columns never used in a season) are left out."""
    gram_c = Cs.T @ Cs
    scale = np.sqrt(np.maximum(np.diag(gram_c), 1e-300))
    std = gram_c / np.outer(scale, scale)
    keep = []
    for c in range(Cs.shape[1]):
        trial = keep + [c]
        if np.diag(gram_c)[c] > 0 and np.linalg.eigvalsh(std[np.ix_(trial, trial)]).min() > 1e-9:
            keep.append(c)
    return Cs[:, keep]


ids_all = np.unique(np.r_[off.ravel(), dfn.ravel()])
J = len(ids_all)
S = []
for s in seasons:
    r = np.flatnonzero(season == s)
    ids = np.unique(np.r_[off[r].ravel(), dfn[r].ravel()])
    ns, Cs = len(ids), independent(C[r])
    m = 2 * ns
    cols = np.column_stack([np.searchsorted(ids, off[r]), ns + np.searchsorted(ids, dfn[r])])
    glob = np.searchsorted(ids_all, ids)
    fs = fold[r]
    G_part = [gram(cols[fs == f], Cs[fs == f], m) for f in range(FOLDS)]
    gy_part = [cross(cols[fs == f], Cs[fs == f], y[r][fs == f], m) for f in range(FOLDS)]
    G_all, gy_all = sum(G_part), sum(gy_part)
    S.append(dict(season=s, r=r, ids=ids, ns=ns, m=m, Cs=Cs, cols=cols, fs=fs, glob=glob,
                  car=np.r_[glob, J + glob],                    # where this season's players sit among career levels
                  poss=np.bincount(cols.ravel(), minlength=m),  # possessions on offense, then on defense
                  G_part=G_part, gy_part=gy_part, G_all=G_all, gy_all=gy_all,
                  gl_train=[cross(cols[fs != f], Cs[fs != f], luck_fold[f][r][fs != f], m) for f in range(FOLDS)],
                  gl_all=cross(cols, Cs, luck_all[r], m)))
print(f"{J:,} players, {sum(d['ns'] for d in S):,} player-seasons")


def season_pieces(d, G, g, lam_season):
    """Everything about one season that does not depend on the career levels: (G + penalty)^-1, the part of X'X
    the season's own effects cannot explain (H), and the same for X'y (u). Solving for the season's own effects
    first and the career levels after is the same as solving everything at once (block elimination), only faster."""
    pen = np.r_[np.full(d["m"], lam_season), np.full(G.shape[0] - d["m"], 1e-8)]   # situation effects: no shrinkage
    Finv = np.linalg.inv(G + np.diag(pen))
    GF = G @ Finv
    return dict(G=G, g=g, Finv=Finv, GF=GF, H=G - GF @ G, u=g - GF @ g)


def career_system(pieces):
    """Normal equations for the career levels once every season's own effects are accounted for."""
    Q, rhs = np.zeros((2 * J, 2 * J)), np.zeros((2 * J, 2))
    for d, p in zip(S, pieces):
        Q[np.ix_(d["car"], d["car"])] += p["H"][:d["m"], :d["m"]]
        rhs[d["car"]] += p["u"][:d["m"]]
    return Q, rhs


def season_effects(d, p, career):
    """Player effects (career level + season deviation; columns = outcome y and the luck correction) and situation
    effects for one season, given the career levels."""
    b = career[d["car"]]
    phi = p["Finv"] @ (p["g"] - p["G"][:, :d["m"]] @ b)
    return b + phi[:d["m"]], phi[d["m"]:]


def solve_career(Q, rhs, lam_career):
    return np.zeros((2 * J, 2)) if np.isinf(lam_career) else np.linalg.solve(Q + lam_career * np.eye(2 * J), rhs)


# ============================================================== cross-validation
def cross_validate():
    """Held-out error of every setting (5 folds by game), and the old method's choice season by season. Prints the
    comparison table and returns the chosen settings and the old method's shrinkage for each season."""
    n_games = len(game_ids)
    err = np.zeros((len(LAM_SEASON), len(LAM_CAREER), len(LUCK)))
    per_game = np.zeros((len(LAM_SEASON), len(LAM_CAREER), len(LUCK), n_games))
    season_err = np.zeros((len(LAM_SEASON), len(S)))        # no career level, no luck taken out: the old method
    base_game = np.zeros(n_games)                            # situation only, no players: for scale
    print("\nCross-validation (5 folds by game) over "
          f"{len(LAM_SEASON)} x {len(LAM_CAREER)} x {len(LUCK)} settings ...", flush=True)
    for f in range(FOLDS):
        for d in S:
            G = d["G_all"] - d["G_part"][f]
            g = d["gy_all"] - d["gy_part"][f]
            a = np.linalg.solve(G[d["m"]:, d["m"]:] + 1e-8 * np.eye(G.shape[0] - d["m"]), g[d["m"]:])
            held = d["fs"] == f
            e = y[d["r"]][held] - d["Cs"][held] @ a
            base_game += np.bincount(game_code[d["r"]][held], e ** 2, minlength=n_games)
    for i, lam_season in enumerate(LAM_SEASON):
        for f in range(FOLDS):
            pieces = [season_pieces(d, d["G_all"] - d["G_part"][f],
                                    np.column_stack([d["gy_all"] - d["gy_part"][f], d["gl_train"][f]]), lam_season)
                      for d in S]
            Q, rhs = career_system(pieces)
            for j, lam_career in enumerate(LAM_CAREER):
                career = solve_career(Q, rhs, lam_career)
                for si, (d, p) in enumerate(zip(S, pieces)):
                    beta, alpha = season_effects(d, p, career)
                    held = d["fs"] == f
                    pred = beta[d["cols"][held]].sum(axis=1) + d["Cs"][held] @ alpha       # rows x 2
                    yy, gc = y[d["r"]][held], game_code[d["r"]][held]
                    for k, luck_share in enumerate(LUCK):
                        e2 = (yy - pred[:, 0] - luck_share * pred[:, 1]) ** 2
                        err[i, j, k] += e2.sum()
                        per_game[i, j, k] += np.bincount(gc, e2, minlength=n_games)
                        if np.isinf(lam_career) and k == 0:
                            season_err[i, si] += e2.sum()
        print(f"  season shrinkage {lam_season:>9,.0f} done", flush=True)

    allowed = np.ones(err.shape, bool)
    for ax, (flag, grid) in enumerate((("--lam-season", LAM_SEASON), ("--lam-career", LAM_CAREER), ("--luck", LUCK))):
        if flag in sys.argv:             # for sensitivity checks: fix a setting by hand; the others are still chosen
            keep = np.zeros(len(grid), bool)
            keep[np.abs(np.log(grid + 1e-12) - np.log(arg(flag, 1.0) + 1e-12)).argmin()] = True
            allowed &= keep.reshape([-1 if a == ax else 1 for a in range(3)])
    best = np.unravel_index(np.where(allowed, err, np.inf).argmin(), err.shape)
    old_lam = LAM_SEASON[season_err.argmin(axis=0)]          # expected.py's choice, season by season
    old_game = np.zeros(n_games)
    for si, d in enumerate(S):                               # the old method's errors, game by game
        in_s = np.unique(game_code[d["r"]])
        old_game[in_s] = per_game[season_err[:, si].argmin(), -1, 0][in_s]

    def row(label, errors_by_game):
        diff = old_game - errors_by_game                     # squared error saved, game by game (games = units)
        gain, se = diff.sum() / n, diff.std() * np.sqrt(n_games) / n
        information = (base_game.sum() - errors_by_game.sum()) / (base_game.sum() - old_game.sum())
        print(f"  {label:58s} {np.sqrt(errors_by_game.sum() / n):.5f}   {gain * 1e4:+7.2f} +/- {1.96 * se * 1e4:5.2f}"
              f"   {information:5.2f}")

    print("\nHow well each version predicts the actual points of possessions in games it was not fitted on")
    print(f"  {'':58s} error     saved vs old (x 1e-4)  information vs old")
    print("  (information = how much better than knowing only the game situation, as a multiple of the old method)")
    row("game situation only, no players", base_game)
    row("OLD: one season at a time, no luck taken out", old_game)
    ic, jc = np.unravel_index(err[:, :-1, 0].argmin(), err[:, :-1, 0].shape)    # best with a career level, no luck out
    row(f"+ career level (season {LAM_SEASON[ic]:,.0f}, career {LAM_CAREER[jc]:,.0f})", per_game[ic, jc, 0])
    il, kl = np.unravel_index(err[:, -1, :].argmin(), err[:, -1, :].shape)      # best with luck out, no career level
    row(f"+ luck taken out (share {LUCK[kl]:.2f}, season {LAM_SEASON[il]:,.0f})", per_game[il, -1, kl])
    row(f"BOTH (season {LAM_SEASON[best[0]]:,.0f}, career {LAM_CAREER[best[1]]:,.0f}, luck share {LUCK[best[2]]:.2f})",
        per_game[best])
    print("  shrinkage = possessions' worth of 'average' each estimate starts from; luck share 1 = all of it taken out")
    on_edge = [nm for nm, ix, grid in (("season", best[0], LAM_SEASON), ("career", best[1], LAM_CAREER))
               if ix in (0, len(grid) - 1 - (nm == "career"))]
    if on_edge:
        print(f"  NOTE: the chosen {', '.join(on_edge)} shrinkage is at the edge of the grid searched; widen the grid.")
    return LAM_SEASON[best[0]], LAM_CAREER[best[1]], LUCK[best[2]], old_lam


# ============================================================== final fit
def final_fit(lam_season, lam_career, luck_share, season_lams=None):
    """Fit on all games; returns per-season player effects (columns off, def as points per 100) and, for every
    player, the possession-weighted average of his seasons with its posterior standard deviation."""
    pieces = [season_pieces(d, d["G_all"], np.column_stack([d["gy_all"], d["gl_all"]]),
                            lam_season if season_lams is None else season_lams[si]) for si, d in enumerate(S)]
    Q, rhs = career_system(pieces)
    career = solve_career(Q, rhs, lam_career)
    out, resid2 = [], 0.0
    for d, p in zip(S, pieces):
        beta, alpha = season_effects(d, p, career)
        b = beta[:, 0] + luck_share * beta[:, 1]
        pred = b[d["cols"]].sum(axis=1) + d["Cs"] @ (alpha[:, 0] + luck_share * alpha[:, 1])
        resid2 += ((y[d["r"]] + luck_share * luck_all[d["r"]] - pred) ** 2).sum()
        out.append((b[:d["ns"]], b[d["ns"]:]))
    sigma2 = resid2 / n
    # Every number is measured against the AVERAGE PLAYER of the same season (possession-weighted), which is what a
    # ranking compares. It also matters for the uncertainty: a shift of every player's level up or down together is
    # the same as a change in that season's league average, which the data cannot tell apart, so only differences
    # from the season's average player are well defined.
    tot = np.zeros(J)
    for d in S:
        np.add.at(tot, d["glob"], d["poss"][:d["ns"]] + d["poss"][d["ns"]:])
    W = np.zeros((J, len(S)))                                  # w[j, s] = share of player j's possessions in season s
    PI = []                                                    # pi[k] = player k's share of season s's possessions
    for si, d in enumerate(S):
        played = d["poss"][:d["ns"]] + d["poss"][d["ns"]:]
        W[d["glob"], si] = played / tot[d["glob"]]
        PI.append(played / played.sum())
    var = {}
    for which, (a_off, a_def) in (("net", (1, -1)), ("off", (1, 0)), ("def", (0, 1))):
        ell = np.zeros((2 * J, J))                             # career part of each player's number
        ell[np.arange(J), np.arange(J)] += a_off
        ell[J + np.arange(J), np.arange(J)] += a_def
        v, own = np.zeros((2 * J, J)), np.zeros(J)
        ell_bar, v_bar = np.zeros((2 * J, len(S))), np.zeros((2 * J, len(S)))
        for si, (d, p) in enumerate(zip(S, pieces)):
            ns, glob, pi, w = d["ns"], d["glob"], PI[si], W[d["glob"], si]
            GF, Fi, k = p["GF"], p["Finv"], np.arange(ns)
            v[np.ix_(d["car"], glob)] += ((a_off * GF[:, :ns] + a_def * GF[:, ns:2 * ns]) * w)[:d["m"]]
            m_bar = np.zeros(GF.shape[0])                      # the season's average player
            m_bar[:ns], m_bar[ns:2 * ns] = a_off * pi, a_def * pi
            ell_bar[glob, si], ell_bar[J + glob, si] = a_off * pi, a_def * pi
            v_bar[d["car"], si] = (GF @ m_bar)[:d["m"]]
            f_bar = Fi @ m_bar
            own[glob] += w ** 2 * (a_off ** 2 * Fi[k, k] + a_def ** 2 * Fi[ns + k, ns + k] + 2 * a_off * a_def * Fi[k, ns + k]
                                   - 2 * (a_off * f_bar[k] + a_def * f_bar[ns + k]) + m_bar @ f_bar)
        if np.isinf(lam_career):
            quad = np.zeros(J)
        else:
            diff = (ell - ell_bar @ W.T) - (v - v_bar @ W.T)
            quad = (diff * np.linalg.solve(Q + lam_career * np.eye(2 * J), diff)).sum(axis=0)
        var[which] = sigma2 * (quad + own) * 1e4               # points per 100, squared
    season_rows, avg = [], np.zeros((J, 2))
    for si, (d, (b_off, b_def)) in enumerate(zip(S, out)):
        ns = d["ns"]
        b_off, b_def = b_off - PI[si] @ b_off, b_def - PI[si] @ b_def      # vs the season's average player
        np.add.at(avg, d["glob"], np.column_stack([b_off, b_def]) * W[d["glob"], si][:, None] * 100)
        season_rows.append(pd.DataFrame({"season": d["season"], "player_id": d["ids"],
                                         "possessions": d["poss"][:ns] + d["poss"][ns:],
                                         "off": b_off * 100, "def": b_def * 100, "net": (b_off - b_def) * 100}))
    seasons_played = pd.concat(season_rows).groupby("player_id").size()
    table = pd.DataFrame({"player_id": ids_all, "seasons": seasons_played.reindex(ids_all).to_numpy(),
                          "possessions": tot, "off": avg[:, 0], "def": avg[:, 1], "impact": avg[:, 0] - avg[:, 1],
                          "off_se": np.sqrt(var["off"]), "def_se": np.sqrt(var["def"]), "se": np.sqrt(var["net"])})
    return table, pd.concat(season_rows), sigma2


if all(flag in sys.argv for flag in ("--lam-season", "--lam-career", "--luck")):
    # every setting given by hand (a quick sensitivity check): no cross-validation, no comparison table
    lam_s, lam_c, share = arg("--lam-season", 1.0), arg("--lam-career", 1.0), arg("--luck", 1.0)
    print("\nAll settings given by hand: no cross-validation.")
else:
    lam_s, lam_c, share, old_lam = cross_validate()
    # the old method, refitted here as a check that the machinery reproduces expected.py
    old_check, old_seasons, _ = final_fit(None, np.inf, 0.0, season_lams=old_lam)
    m_ = old_seasons.merge(old_effects, on=["season", "player_id"], suffixes=("", "_saved"))
    m_["pw"] = m_["possessions"] * m_["net_saved"]
    season_mean = m_.groupby("season")["pw"].transform("sum") / m_.groupby("season")["possessions"].transform("sum")
    gap = np.abs(m_["net"] - (m_["net_saved"] - season_mean)).max()      # both against the season's average player
    print(f"\nCheck: refitting the old method here reproduces expected.py's player effects "
          f"(largest difference {gap:.1e} per 100 over {len(m_):,} player-seasons)")
    if gap > 0.01:
        print("  WARNING: the old method was not reproduced; the comparison above is not like for like.")

new, new_seasons, sigma2 = final_fit(lam_s, lam_c, share)
tau = lambda lam: np.sqrt(sigma2 / lam) * 100
print(f"\nChosen: season shrinkage {lam_s:,.0f}, career shrinkage {lam_c:,.0f}, share of luck taken out {share:.2f}")
if np.isinf(lam_c):
    print(f"  no career level (each season on its own): spread of players' true levels {tau(lam_s):.2f} per 100 "
          f"(offense or defense); noise per possession {np.sqrt(sigma2):.3f} points")
else:
    print(f"  = spread of players' true career levels {tau(lam_c):.2f} per 100 (offense or defense), of a season around "
          f"the career level {tau(lam_s):.2f}; noise per possession {np.sqrt(sigma2):.3f} points")


# ============================================================== old vs new, with rank ranges
def season_average(table, col, se_col):
    """players.py's old season average: each season weighted by 1 / se^2."""
    t = table.dropna(subset=[col, se_col])
    t = t[t[se_col] > 0]
    w = 1 / t[se_col] ** 2
    g = t.assign(w=w, vw=t[col] * w).groupby("player_id")
    return pd.DataFrame({"seasons": g.size(), "value": g["vw"].sum() / g["w"].sum(), "se": 1 / np.sqrt(g["w"].sum())})


def rank_ranges(value, se):
    """players.py's rule: draw every player's value from what is known about it, rank the draws, among the
    better-measured half of players (standard error at most the median)."""
    value, se = np.asarray(value), np.asarray(se)
    measured = se <= np.median(se)
    draw = np.random.default_rng(11)
    sims = value[measured, None] + se[measured, None] * draw.standard_normal((measured.sum(), 1000))
    ranks = (-sims).argsort(axis=0).argsort(axis=0) + 1
    out = np.full(len(value), "-", dtype=object)
    out[measured] = [f"{int(a)}-{int(b)}" for a, b in np.quantile(ranks, [0.025, 0.975], axis=1).T]
    width = np.full(len(value), np.nan)
    width[measured] = np.diff(np.quantile(ranks, [0.025, 0.975], axis=1), axis=0)[0]
    return out, width


old = season_average(old_effects, "net", "net_se")
old["range"], old["width"] = rank_ranges(old["value"], old["se"])
new["range"], new["width"] = rank_ranges(new["impact"], new["se"])
new["player"] = names.reindex(new["player_id"]).to_numpy()
both = new.set_index("player_id").join(old[["value", "se", "range", "width"]].add_prefix("old_"))
both = both.sort_values("impact", ascending=False)
z_old, z_new = old["value"] / old["se"], both["impact"] / both["se"]
print(f"\nOVERALL IMPACT, old vs new (net per 100; +/- = 95% margin; rank range among the better-measured half)")
print(f"  clearly above / below average (95%): old {(z_old > 1.96).sum()} / {(z_old < -1.96).sum()}, "
      f"new {(z_new > 1.96).sum()} / {(z_new < -1.96).sum()}")
print(f"  median 95% margin among the better-measured half: old {1.96 * old['se'][old['se'] <= old['se'].median()].median():.2f}, "
      f"new {1.96 * both['se'][both['se'] <= both['se'].median()].median():.2f}")
top = both.head(20)
show = pd.DataFrame({"player": top["player"], "seasons": top["seasons"],
                     "old": top["old_value"].map("{:+.2f}".format), "old +/-": (1.96 * top["old_se"]).map("{:.2f}".format),
                     "old rank range": top["old_range"],
                     "new": top["impact"].map("{:+.2f}".format), "new +/-": (1.96 * top["se"]).map("{:.2f}".format),
                     "new rank range": top["range"]})
print(show.to_string(index=False))

# ============================================================== save
if "--lam-season" in sys.argv and "--lam-career" in sys.argv and "--luck" in sys.argv and "--save" not in sys.argv:
    raise SystemExit("\nSettings given by hand: nothing saved (add --save to keep them).")
con = sqlite3.connect(DB)
new.drop(columns=["range", "width", "player"]).to_sql("player_impact", con, if_exists="replace", index=False)
new_seasons.to_sql("player_season_impact", con, if_exists="replace", index=False)
pd.DataFrame({"name": ["lam_season", "lam_career", "luck_share", "sigma2"],
              "value": [lam_s, lam_c, share, sigma2]}).to_sql("impact_settings", con, if_exists="replace", index=False)
con.commit()
con.close()
print("\nSaved player_impact, player_season_impact, impact_settings.  Next: python momentum.py")
