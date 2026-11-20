"""
NBA Momentum Study - Step 3c: expected points for every possession.

"Momentum" is always measured against what a possession was EXPECTED to be
worth, so this expectation has to be fair. Four versions, each adding
knowledge:

  exp_base          the two teams' quality that season, home court, score,
                    period and clock
  exp_chain         ... plus how the possession started (after a steal, a
                    miss, a made basket, a timeout ...) = the "chain"
  exp_lineup        the ten PLAYERS on the floor instead of team quality
  exp_lineup_chain  players + how the possession started

How it is built, and why (nothing here is a hand-picked number):

 * Ridge regression. Each team (or each player, on offense and on defense)
   gets its own effect, estimated together with everyone else's. With
   hundreds of players some have few possessions, so effects are shrunk
   toward zero; this is the standard "regularized adjusted plus-minus"
   approach for separating a player from his teammates and opponents. HOW
   MUCH to shrink is not chosen by me: it is the amount that predicts best
   on games left out of the fit (5-fold cross-validation, by game).

 * Smooth curves for the ordinary run of play. Score margin and clock enter
   through restricted cubic splines (smooth curves) with knots at fixed
   percentiles of the data (the standard 5-knot placement), plus a separate
   margin curve for the 4th quarter/overtime, when big leads change who
   plays.

 * Game states that the smooth curves miss. Checking where the expectation
   went wrong showed four situations in which possessions are worth clearly
   more or less than the curves say, and each lasts for a stretch of play,
   so it could pass for momentum. They are added as their own terms, with
   boundaries taken from the rules of the game, not from the results:
     - the last seconds of a period (heaves, 2-for-1s): 0-5, 6-24 (the shot
       clock is off) and 25-60 seconds left, in the 1st-3rd quarters;
     - the end of the game (intentional fouls, playing for one shot): in the
       last six minutes of the 4th quarter and overtime, the score margin
       counted in possessions (tied; 1-3, 4-6, 7-9, 10+ ahead or behind)
       crossed with the time left (0-24 s, 25-60 s, 1-2 min, 2-6 min);
     - the defense in the foul penalty, so every further foul gives free
       throws (from the team-foul counts written in the play-by-play, e.g.
       "S.FOUL (P1.T4)" = that team's 4th foul of the period);
     - days of rest for each team (back-to-back, two days or more, first
       game of the season; one day off is the reference).

 * Cross-fitting. The expectation for a possession always comes from a model
   fitted WITHOUT that game's fold, so a team's hot night cannot leak into
   its own expectation.

 * Playoffs are predicted from that season's regular-season fit.

Also saves `player_effects`: each player's estimated effect on points per
100 possessions on offense and defense, per season, with its uncertainty
(used by players.py), from the model without the chain (a steal that starts
a fast break should count for the player).

Run after build_lineups.py.   Usage:  python expected.py [--db data/fake_nba.db]
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "data/nba.db"
FOLDS = 5
LAMBDAS = np.logspace(0, 5, 21)                 # candidate amounts of shrinkage; cross-validation picks
KNOT_PCT = [5, 27.5, 50, 72.5, 95]              # standard 5-knot placement for restricted cubic splines

con = sqlite3.connect(DB)
poss = pd.read_sql_query(
    """SELECT p.*, l.o1, l.o2, l.o3, l.o4, l.o5, l.d1, l.d2, l.d3, l.d4, l.d5, l.complete
       FROM poss p LEFT JOIN lineups l USING (game_id, poss_num)
       WHERE p.points_ok = 1 ORDER BY p.game_id, p.poss_num""", con)
if "playoffs" not in poss:
    poss["playoffs"] = 0
n = len(poss)
y = poss["points"].to_numpy(float)
season = poss["season"].to_numpy()
regular = (poss["playoffs"] == 0).to_numpy()


def spline(x, knots):
    """Restricted cubic spline basis: a smooth curve that is a straight line beyond the outer knots."""
    k = np.asarray(knots, float)
    cube = lambda v: np.maximum(v, 0) ** 3
    scale = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(len(k) - 2):
        cols.append((cube(x - k[j]) - cube(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + cube(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / scale)
    return np.column_stack(cols)


def knots_of(x):
    k = np.unique(np.percentile(x, KNOT_PCT))
    return k if len(k) >= 3 else np.array([x.min(), np.median(x), x.max() + 1e-9])


margin = poss["start_margin"].to_numpy(float)                 # offense's lead at the start of the possession
clock = poss["seconds_left"].to_numpy(float)
period = poss["period"].to_numpy()
log_clock = np.log1p(clock)
late = period >= 4


def in_penalty():
    """1 if the defense is already in the foul penalty when the possession starts, from the team-foul counts in the
    play-by-play: "(P2.T4)" = the fouler's 2nd personal foul and his team's 4th of the period, "(P3.PN)" = a foul
    committed while already in the penalty. A team is in the penalty after its 4th team foul of a quarter (3rd in
    overtime), and in the last two minutes of a period after its first foul in those two minutes."""
    f = pd.read_sql_query("""SELECT game_id, period, seconds_left, player1_team_abbreviation AS team,
                                    COALESCE(homedescription, '') || ' ' || COALESCE(visitordescription, '') AS txt
                             FROM events WHERE eventmsgtype = 6""", con)
    note = f["txt"].str.extract(r"\(P\d+\.(?:T(\d+)|(PN))\)")
    count = pd.to_numeric(note[0], errors="coerce").to_numpy()
    limit = np.where(f["period"] <= 4, 4, 3)
    into = (note[1] == "PN").to_numpy() | (count >= limit) | ((count >= 1) & (f["seconds_left"] <= 120).to_numpy())
    went_in = f[into].groupby(["game_id", "period", "team"])["seconds_left"].max()     # clock when it began (counts down)
    began = went_in.reindex(pd.MultiIndex.from_frame(poss[["game_id", "period", "defense"]])).to_numpy()
    return began > clock                                                               # strictly before this possession


def rest_days():
    """Days since each team's previous game (regular season and playoffs in date order), for the offense and the
    defense: back-to-back, 2+ days, first game of the season (1 day off = the reference)."""
    g = poss.drop_duplicates("game_id")[["game_id", "season", "gamedate", "home", "away"]]
    long = pd.concat([g.assign(team=g["home"]), g.assign(team=g["away"])])
    long = long.assign(date=pd.to_datetime(long["gamedate"])).sort_values(["season", "team", "date", "game_id"])
    days = long.groupby(["season", "team"])["date"].diff().dt.days
    days.index = pd.MultiIndex.from_frame(long[["game_id", "team"]])
    cols = []
    for side in ("offense", "defense"):
        d = days.reindex(pd.MultiIndex.from_frame(poss[["game_id", side]])).to_numpy()
        cols += [d == 1, d >= 3, np.isnan(d)]
    return np.column_stack(cols)


# end of the game: margin in possessions x time left, in the last six minutes of the 4th quarter and overtime
margin_bin = np.digitize(margin, [-9.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 9.5])   # behind 10+, 7-9, 4-6, 1-3, tied, ahead 1-3 ...
time_band = np.digitize(clock, [24.5, 60.5, 120.5])                            # 0-24 s, 25-60 s, 1-2 min, 2-6 min
end_game = np.where(late & (clock <= 360), margin_bin * 4 + time_band, -1)
# end of the 1st-3rd quarters: 0-5 s, 6-24 s, 25-60 s left
end_quarter = np.where((period <= 3) & (clock <= 60), np.digitize(clock, [5.5, 24.5]), -1)
situation = np.column_stack([
    np.ones(n), (poss["is_home"] == 1).to_numpy(), pd.get_dummies(np.minimum(poss["period"], 5)).to_numpy(float)[:, 1:],
    spline(margin, knots_of(margin)), spline(margin, knots_of(margin[late])) * late[:, None], late,
    spline(log_clock, knots_of(log_clock)),
    end_game[:, None] == np.arange(36), end_quarter[:, None] == np.arange(3), in_penalty(), rest_days()]).astype(float)
print(f"game states: defense in the foul penalty for {situation[:, -7].mean():.1%} of possessions; "
      f"last six minutes of the 4th quarter or overtime {(end_game >= 0).mean():.1%}")
chain = pd.get_dummies(poss["starttype"].fillna("?")).to_numpy(float)[:, 1:]


def fit_layer(rows_fit, rows_all, off_idx, def_idx, n_ent, C, want_effects=False):
    """Ridge fit for one season. off_idx/def_idx: (rows, slots) entity numbers. Returns cross-fitted
    predictions for rows_fit, full-fit predictions for the other rows, and optionally the entity effects."""
    q = C.shape[1]
    p = 2 * n_ent + q
    slots = off_idx.shape[1]
    cols_all = np.column_stack([off_idx, def_idx + n_ent])            # entity columns of the design matrix

    def stats(r):
        # X'X and X'y without ever building the (huge, mostly empty) design matrix
        A = np.zeros((p, p))
        idx, Cr, yr = cols_all[r], C[r], y[r]
        m = 2 * n_ent
        pairs = np.zeros(m * m)
        block = np.zeros((m, q))
        b_ = np.zeros(p)
        for a in range(2 * slots):
            for b in range(2 * slots):
                pairs += np.bincount(idx[:, a] * m + idx[:, b], minlength=m * m)
            for c in range(q):
                block[:, c] += np.bincount(idx[:, a], weights=Cr[:, c], minlength=m)
            b_[:m] += np.bincount(idx[:, a], weights=yr, minlength=m)
        A[:m, :m] = pairs.reshape(m, m)
        A[:m, m:], A[m:, :m] = block, block.T
        A[m:, m:] = Cr.T @ Cr
        b_[m:] = Cr.T @ yr
        return A, b_

    def predict(beta, r):
        return beta[cols_all[r]].sum(axis=1) + C[r] @ beta[2 * n_ent:]

    game = poss["game_id"].to_numpy()[rows_fit]
    fold = pd.factorize(game)[0] % FOLDS
    parts = [stats(rows_fit[fold == f]) for f in range(FOLDS)]
    A_all, b_all = sum(a for a, _ in parts), sum(b for _, b in parts)
    penalty = np.r_[np.ones(2 * n_ent), np.full(q, 1e-8)]

    def cv_error(lam):
        err = 0.0
        for f in range(FOLDS):
            beta = np.linalg.solve(A_all - parts[f][0] + np.diag(lam * penalty), b_all - parts[f][1])
            r = rows_fit[fold == f]
            err += ((y[r] - predict(beta, r)) ** 2).sum()
        return err

    errors = np.array([cv_error(lam) for lam in LAMBDAS])
    lam = LAMBDAS[errors.argmin()]
    pred = np.full(len(rows_all), np.nan)
    for f in range(FOLDS):                                           # cross-fitted predictions
        beta = np.linalg.solve(A_all - parts[f][0] + np.diag(lam * penalty), b_all - parts[f][1])
        r = rows_fit[fold == f]
        pred[np.searchsorted(rows_all, r)] = predict(beta, r)
    inv = np.linalg.inv(A_all + np.diag(lam * penalty))
    beta = inv @ b_all
    rest = np.setdiff1d(rows_all, rows_fit)
    if len(rest):
        pred[np.searchsorted(rows_all, rest)] = predict(beta, rest)   # e.g. playoffs: from the regular-season fit
    if not want_effects:
        return pred, lam, None
    sigma2 = ((y[rows_fit] - predict(beta, rows_fit)) ** 2).mean()
    return pred, lam, (beta[:2 * n_ent], sigma2 * inv[:2 * n_ent, :2 * n_ent])


out = {name: np.full(n, np.nan) for name in ("exp_base", "exp_chain", "exp_lineup", "exp_lineup_chain")}
effects = []
has_lineup = (poss["complete"] == 1).to_numpy()
players_off = poss[[f"o{k}" for k in range(1, 6)]].to_numpy()
players_def = poss[[f"d{k}" for k in range(1, 6)]].to_numpy()
print(f"{n:,} possessions; complete lineups for {has_lineup.mean():.1%}\n")
print("season | shrinkage chosen by cross-validation: teams, teams+chain, players, players+chain")
for s in np.unique(season):
    in_s = season == s
    # --- teams
    rows_all = np.flatnonzero(in_s)
    rows_fit = np.flatnonzero(in_s & regular)
    teams, t_off = np.unique(poss["offense"].to_numpy()[rows_all], return_inverse=True)
    t_def = pd.Series(np.arange(len(teams)), index=teams).reindex(poss["defense"].to_numpy()[rows_all]).fillna(0).to_numpy().astype(int)
    oi, di = np.zeros((n, 1), int), np.zeros((n, 1), int)
    oi[rows_all, 0], di[rows_all, 0] = t_off, t_def
    lams = []
    for name, C in (("exp_base", situation), ("exp_chain", np.column_stack([situation, chain]))):
        pred, lam, _ = fit_layer(rows_fit, rows_all, oi, di, len(teams), C)
        out[name][rows_all] = pred
        lams.append(lam)
    # --- players
    rows_all = np.flatnonzero(in_s & has_lineup)
    rows_fit = np.flatnonzero(in_s & has_lineup & regular)
    if len(rows_fit):
        ids = np.unique(np.r_[players_off[rows_fit].ravel(), players_def[rows_fit].ravel()])
        number = pd.Series(np.arange(1, len(ids) + 1), index=ids)        # 0 = a player not seen in the regular season
        oi = number.reindex(players_off.ravel()).fillna(0).to_numpy().astype(int).reshape(n, 5)
        di = number.reindex(players_def.ravel()).fillna(0).to_numpy().astype(int).reshape(n, 5)
        n_ent = len(ids) + 1
        for name, C, keep in (("exp_lineup", situation, True), ("exp_lineup_chain", np.column_stack([situation, chain]), False)):
            pred, lam, eff = fit_layer(rows_fit, rows_all, oi, di, n_ent, C, want_effects=keep)
            out[name][rows_all] = pred
            lams.append(lam)
            if keep:
                beta, cov = eff
                o, d = np.arange(1, n_ent), np.arange(n_ent + 1, 2 * n_ent)
                counts = np.bincount(oi[rows_fit].ravel(), minlength=n_ent)[1:]
                effects.append(pd.DataFrame({
                    "season": s, "player_id": ids, "off_poss": counts,
                    "off": beta[o] * 100, "def": beta[d] * 100,                 # points per 100 possessions
                    "off_se": np.sqrt(np.diag(cov)[o]) * 100, "def_se": np.sqrt(np.diag(cov)[d]) * 100,
                    "net": (beta[o] - beta[d]) * 100,
                    "net_se": np.sqrt(np.diag(cov)[o] + np.diag(cov)[d] - 2 * cov[o, d]) * 100}))
    print(f"  {s} | " + ", ".join(f"{v:,.0f}" for v in lams), flush=True)

pd.DataFrame({"game_id": poss["game_id"], "poss_num": poss["poss_num"], **out}).to_sql(
    "poss_expected", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_pe ON poss_expected(game_id, poss_num)")
if effects:
    pd.concat(effects).to_sql("player_effects", con, if_exists="replace", index=False)
con.commit()
con.close()

print("\nHow well each version predicts a possession it was not fitted on (lower error = better):")
for name, v in out.items():
    m = regular & ~np.isnan(v) & ~np.isnan(out["exp_lineup"])
    print(f"  {name:17s} error {np.sqrt(((y[m] - v[m]) ** 2).mean()):.5f}   (guessing the league average: {y[m].std():.5f})")
print("Next: python momentum.py")
