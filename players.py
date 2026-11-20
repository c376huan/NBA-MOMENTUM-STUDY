"""
NBA Momentum Study - Step 5: which players create momentum, and which stop it?

Run after momentum.py. Three different ways of answering, side by side,
because they mean different things. Unit: points per 100 possessions.
"Momentum against" a team = the opponent's recent scoring rate minus its own
(from momentum.py), measured in standard deviations so that 1 = a typical
swing. Nothing uses a hand-picked cut-off or a minimum number of games.

 VIEW 1 - ON THE FLOOR (the fairest one: teammates and opponents accounted for)
   Every possession is explained by the ten players on the floor at once
   (ridge regression, the "regularized adjusted plus-minus" method; the
   amount of shrinkage is chosen by cross-validation, i.e. by what predicts
   games left out of the fit). A player gets:
     STARTS IT   his effect on his team's scoring margin whenever he plays
                 (offense plus defense). This is how momentum gets created
                 from a standing start. One estimate per season.
     STOPS IT    how much MORE he helps when momentum is against his team,
                 split into scoring (his team's offense) and defense (the
                 opponent's offense), beyond his usual level when he plays.
     KEEPS IT    how much MORE he helps when his team already has momentum.
                 STOPS and KEEPS are one estimate per player from all his
                 seasons together (a single season is too little data). His
                 usual level is taken out first (see interaction_model).
   Small samples are pulled toward zero automatically, so no minimum is
   needed, and the estimates are already "fair".

 VIEW 2 - CREDIT FOR THE PLAYS (easy to read; rewards being involved)
   On possessions he finishes (shot, free throws, turnover) or assists on,
   and on defensive possessions he ends (steal, block, defensive rebound):
   how many points better than expected the possession went, per 100
   possessions he is on the floor. Misses and turnovers count against him.
     CREDIT      over all his possessions
     CREDIT WHEN MOMENTUM IS AGAINST
                 the same, counted more the more momentum was against his team
     LIFT        the second minus the first: does he do more than usual when
                 momentum is against? (compared with the average player's
                 lift, because weighting by momentum lowers everyone's number)
   The first two are INVOLVEMENT measures, not momentum measures: assists are
   only recorded on made baskets, and steals, blocks and defensive rebounds
   only on stops, so credit is positive by construction, and a player's
   assist, defensive-rebound, steal and block rates explain about 85% of it.
   Only LIFT, a comparison of a player with himself, is about momentum.

 VIEW 3 - WHAT HAPPENS AFTER HIS PLAYS
   His team's momentum from the next possession on (chain and players on
   the floor removed) after his dunks, blocks, steals-and-scores and
   and-ones (CREATES), and after his baskets counted more the more momentum
   was against (STOPS), minus his own everyday baseline.

Seasons: every number is a RATE worked out season by season and then
averaged, each season counting according to how precisely it is measured
(how much he played) - so a long career gets no head start and a season of
a few minutes does not count like a full one.
Uncertainty: "+/-" is the 95% margin of error. Luck is measured with games
as the unit, pooled over all players, so a handful of plays can never look
precise. Lists in views 2 and 3 are ranked by the fair estimate (pulled
toward average according to how much of the spread between players is luck).
Every player is tagged:
     clearly above average / clearly below average / can't tell from average
"Repeats?" = does a player's number in one season line up with his next
season (correlation; near 0 = luck, clearly above 0 = a lasting trait).

Usage:
    python players.py [--db data/fake_nba.db] [--top 12]
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
TOP = arg("--top", 12)
FOLDS = 5
LAMBDAS = np.logspace(2, 8, 13)          # candidate amounts of shrinkage; cross-validation picks
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)

# ============================================================== load data
con = sqlite3.connect(DB)
# make the database use the real (game, possession) indexes when joining tables; left to itself it
# sometimes builds a makeshift index on half the key, which makes loading take many minutes
con.execute("PRAGMA automatic_index = OFF")
try:
    poss = pd.read_sql_query(
        """SELECT p.game_id, p.poss_num, p.season, p.playoffs, p.is_home, p.home, p.points, p.fg2m, p.fg3m,
                  l.o1, l.o2, l.o3, l.o4, l.o5, l.d1, l.d2, l.d3, l.d4, l.d5,
                  m.rate_raw, m.rate_lineup, m.before_off, m.before_def, m.ahead_lineup, m.ahead_lineup_lo, m.ahead_lineup_hi
           FROM poss p JOIN poss_momentum m USING (game_id, poss_num) JOIN lineups l USING (game_id, poss_num)
           ORDER BY p.game_id, p.poss_num""", con)
    effects = pd.read_sql_query("SELECT * FROM player_effects", con)
    names = pd.read_sql_query("SELECT * FROM players", con).drop_duplicates("player_id").set_index("player_id")["player"]
    settings = pd.read_sql_query("SELECT * FROM momentum_settings", con).set_index("name")["value"]
    H, H_LO, H_HI = (float(settings[k]) for k in ("half_life", "half_life_lo", "half_life_hi"))
except Exception as e:
    raise SystemExit(f"Run build_lineups.py, expected.py and momentum.py first. [{e}]")
ev = pd.read_sql_query(
    """SELECT game_id, poss_num, eventmsgtype AS etype, homedescription AS home_txt, visitordescription AS away_txt,
              player1_id AS p1, player1_team_abbreviation AS t1, player2_id AS p2, player3_id AS p3
       FROM events WHERE poss_num IS NOT NULL AND eventmsgtype <= 5""", con)
con.close()

n = len(poss)
game = poss["game_id"].to_numpy()
season = poss["season"].to_numpy()
playoffs = (poss["playoffs"] == 1).to_numpy()
sign = np.where(poss["is_home"] == 1, 1, -1)
off_players = poss[[f"o{k}" for k in range(1, 6)]].to_numpy()
def_players = poss[[f"d{k}" for k in range(1, 6)]].to_numpy()
has_lineup = poss["rate_lineup"].notna().to_numpy()
rate_raw, rate_lineup = poss["rate_raw"].to_numpy(), np.nan_to_num(poss["rate_lineup"].to_numpy())
momentum = (poss["before_off"] - poss["before_def"]).to_numpy()        # of the team with the ball
SD = momentum[~playoffs].std()
against = np.maximum(-momentum, 0) / SD        # momentum against the OFFENSE, in standard deviations
ahead = np.maximum(momentum, 0) / SD           # momentum the offense has (= against the DEFENSE)
one = np.ones(n)

row_of = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
ev["i"] = row_of.reindex(pd.MultiIndex.from_frame(ev[["game_id", "poss_num"]].astype("int64"))).to_numpy()
ev = ev.dropna(subset=["i"]).astype({"i": int}).reset_index(drop=True)
text = ev["home_txt"].fillna("") + " | " + ev["away_txt"].fillna("")
ei = ev["i"].to_numpy()
pid1, pid2, pid3 = (ev[c].fillna(0).to_numpy().astype("int64") for c in ("p1", "p2", "p3"))
p1_side = np.where(ev["t1"].to_numpy() == poss["home"].to_numpy()[ei], 1, -1)
has_team1 = ev["t1"].notna().to_numpy()
etype = ev["etype"].to_numpy()


# ============================================================== helpers
def fair(value, se):
    """Pull each estimate toward the average according to how much of the spread is luck (empirical Bayes)."""
    w = 1 / se ** 2
    centre = np.sum(w * value) / np.sum(w)
    real_var = max(np.sum(w * (value - centre) ** 2) / np.sum(w) - np.sum(w * se ** 2) / np.sum(w), 0.0)
    return centre + real_var / (real_var + se ** 2) * (value - centre), centre, np.sqrt(real_var)


def repeats(table, col):
    """Correlation between a player's number in one season and his next season."""
    t = table[["player_id", "season", col]].dropna()
    pairs = t.merge(t.assign(season=t["season"] - 1), on=["player_id", "season"], suffixes=("", "_next"))
    if len(pairs) < 30:
        return "not enough seasons to say"
    a, b = pairs[col].to_numpy(), pairs[f"{col}_next"].to_numpy()
    r = np.corrcoef(a, b)[0, 1]
    luck = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(2000)])
    verdict = "a lasting trait" if r > np.quantile(luck, 0.99) else "weak hint" if r > np.quantile(luck, 0.95) else "no - looks like luck"
    return f"{r:+.2f} over {len(pairs):,} pairs of seasons ({verdict})"


def season_average(table, col, se_col):
    """Average of a player's seasons, each counting by how precisely it is measured (inverse-variance weights)."""
    t = table.dropna(subset=[col, se_col])
    t = t[t[se_col] > 0]
    w = 1 / t[se_col] ** 2
    g = t.assign(w=w, vw=t[col] * w).groupby("player_id")
    return pd.DataFrame({"seasons": g.size(), "value": g["vw"].sum() / g["w"].sum(), "se": 1 / np.sqrt(g["w"].sum())})


def present(title, avg, repeat_text, extra=(), note="", already_fair=False, nobody=False):
    """Print one ranked list. avg: one row per player with value and se. nobody=True: the data cannot tell players
    apart on this measure, so nobody is flagged and the order means nothing."""
    avg = avg.copy()
    if already_fair:                       # ridge estimates are already pulled toward the average player (zero)
        avg["fair estimate"], centre, spread_text = avg["value"], 0.0, ""
    else:
        avg["fair estimate"], centre, real_sd = fair(avg["value"].to_numpy(), avg["se"].to_numpy())
        spread_text = f" league average {centre:+.2f}; real spread between players {real_sd:.2f};"
    z = (avg["value"] - centre) / avg["se"]
    avg["verdict"] = np.select([z > 1.96, z < -1.96], ["clearly above average", "clearly below average"], "can't tell from average")
    if nobody:
        avg["verdict"] = "can't tell from average"
    avg["player"] = names.reindex(avg.index).to_numpy()
    avg["+/-"] = 1.96 * avg["se"]
    avg["evidence"] = z                    # when fair estimates tie (no real spread), order by strength of evidence
    avg = avg.sort_values(["fair estimate", "evidence"], ascending=False)
    above, below = (avg["verdict"] == "clearly above average").sum(), (avg["verdict"] == "clearly below average").sum()
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  Repeats?  {repeat_text}")
    if nobody:
        print("  The data cannot tell players apart on this (see above), so nobody is flagged and the order below means nothing.")
    print(f"  {len(avg):,} players;{spread_text} clearly above average: {above}, clearly below: {below}"
          + ("" if already_fair else f" (luck alone would flag about {0.025 * len(avg):.0f} each way)"))
    cols = ["player", "seasons", *extra, "value", "+/-", "fair estimate", "verdict"]
    fmt = {"value": "{:+.2f}".format, "+/-": "{:.2f}".format, "fair estimate": "{:+.2f}".format,
           **{c: "{:+.2f}".format for c in extra}}
    print(avg[cols].head(TOP).to_string(index=False, formatters=fmt))
    print("  ...")
    print(avg[cols].tail(5).to_string(index=False, header=False, formatters=fmt))
    return avg.reset_index().rename(columns={"index": "player_id"}).assign(measure=title.split(" - ")[0])[["measure", "player_id", *cols]]


def ridge_stats(cols, vals, y, rows, m):
    """X'X and X'y for the given rows without building the (huge, mostly empty) design matrix X.
    cols[i, s] = the column that slot s of row i fills, vals[i, s] = the value it puts there."""
    c_, v_, y_ = cols[rows], vals[rows], y[rows]
    upper, diag, b = np.zeros(m * m), np.zeros(m), np.zeros(m)
    for s1 in range(c_.shape[1]):
        diag += np.bincount(c_[:, s1], weights=v_[:, s1] ** 2, minlength=m)
        b += np.bincount(c_[:, s1], weights=v_[:, s1] * y_, minlength=m)
        for s2 in range(s1 + 1, c_.shape[1]):
            upper += np.bincount(c_[:, s1] * m + c_[:, s2], weights=v_[:, s1] * v_[:, s2], minlength=m * m)
    A = upper.reshape(m, m)
    A = A + A.T
    A[np.diag_indices(m)] += diag
    return A, b


def ridge_cv(cols, vals, y, fold, m, lambdas):
    """Ridge fits on 4 folds, judged on the 5th, for every candidate shrinkage. Returns the per-fold X'X and X'y,
    their totals, the held-out error for each shrinkage, and the held-out error of predicting 0."""
    parts = [ridge_stats(cols, vals, y, np.flatnonzero(fold == k), m) for k in range(FOLDS)]
    A_all, b_all = sum(p[0] for p in parts), sum(p[1] for p in parts)
    eye = np.eye(m)
    errors, nothing = np.zeros(len(lambdas)), 0.0
    for k in range(FOLDS):
        A_train, b_train = A_all - parts[k][0], b_all - parts[k][1]
        rows_k = np.flatnonzero(fold == k)
        nothing += (y[rows_k] ** 2).sum()
        for li, lam in enumerate(lambdas):
            beta = np.linalg.solve(A_train + lam * eye, b_train)
            errors[li] += ((y[rows_k] - (beta[cols[rows_k]] * vals[rows_k]).sum(axis=1)) ** 2).sum()
    return parts, A_all, b_all, errors, nothing


def interaction_model(r):
    """VIEW 1, STOPS and KEEPS. What is left of each possession after the chain and the players' usual effects,
    explained by which players are on the floor TIMES how much momentum is against (or for) their team. Four
    numbers per player: offense/against, offense/for, defense/against, defense/for.

    Two steps, because the expected-points model shrinks every player's usual effect toward zero: a good
    player's possessions therefore still come out a little above expectation whenever he plays, and a
    momentum term alone would pick that up and call it "helps when momentum is against" (the first version
    did: its lists were mostly the plus-minus ranking again).
      1. Each player's left-over level when he is on the floor (offense and defense), fitted by ridge with the
         shrinkage chosen by cross-validation, cross-fitted by game fold, and taken out of every possession.
      2. The momentum terms on what is left, shrinkage chosen by cross-validation over games. "Players differ"
         only if this predicts games left out of the fit better than step 1 alone.
    Uncertainty: the sampling spread of the shrunk estimates (sigma^2 M X'X M, M = (X'X + lambda I)^-1), not the
    width of the shrinkage itself. A split of the games into two halves checks whether the numbers repeat."""
    ids = np.unique(np.r_[off_players[r].ravel(), def_players[r].ravel()])
    P = len(ids)
    number = pd.Series(np.arange(P), index=ids)
    o = number.reindex(off_players[r].ravel()).to_numpy().reshape(-1, 5)
    d = number.reindex(def_players[r].ravel()).to_numpy().reshape(-1, 5)
    a, f = against[r], ahead[r]
    C = np.column_stack([np.ones(len(r)), a, f])            # league-wide effect of momentum: taken out first
    y = rate_lineup[r]
    y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    fold = pd.factorize(game[r])[0] % FOLDS

    # step 1: each player's left-over level, cross-fitted
    cols1, vals1 = np.column_stack([o * 2, d * 2 + 1]), np.ones((len(r), 10))
    parts1, A1, b1, err1, zero1 = ridge_cv(cols1, vals1, y, fold, 2 * P, LAMBDAS)
    lam1 = LAMBDAS[int(np.argmin(err1))]
    level = np.zeros(len(r))
    for k in range(FOLDS):
        rows_k = np.flatnonzero(fold == k)
        beta1 = np.linalg.solve(A1 - parts1[k][0] + lam1 * np.eye(2 * P), b1 - parts1[k][1])
        level[rows_k] = beta1[cols1[rows_k]].sum(axis=1)
    y2 = y - level

    # step 2: the momentum terms
    is_against = (a > 0)[:, None]
    cols = np.column_stack([o * 4 + np.where(is_against, 0, 1), d * 4 + np.where(is_against, 3, 2)])
    vals = np.repeat((a + f)[:, None], 10, axis=1)          # the same for all ten players of a possession
    m = 4 * P
    parts, A_all, b_all, errors, nothing = ridge_cv(cols, vals, y2, fold, m, LAMBDAS)
    li = int(np.argmin(errors))
    lam = LAMBDAS[li]
    eye = np.eye(m)
    M = np.linalg.inv(A_all + lam * eye)
    beta = M @ b_all
    sigma2 = ((y2 - (beta[cols] * vals).sum(axis=1)) ** 2).mean()
    k4 = np.arange(P) * 4

    def contrast(i, j):
        """Sampling variance of beta_i - beta_j: sigma^2 x (M X'X M) = sigma^2 x (M - lambda M M)."""
        Mi, Mj = M[i], M[j]
        cov = lambda u, v, Mu, Mv: M[u, v] - lam * (Mu * Mv).sum(axis=1)
        return sigma2 * (cov(i, i, Mi, Mi) + cov(j, j, Mj, Mj) - 2 * cov(i, j, Mi, Mj))

    out = pd.DataFrame({
        "stop_scoring": beta[k4], "stop_defense": -beta[k4 + 2],
        "stops": beta[k4] - beta[k4 + 2], "stops_se": np.sqrt(contrast(k4, k4 + 2)),
        "keeps": beta[k4 + 1] - beta[k4 + 3], "keeps_se": np.sqrt(contrast(k4 + 1, k4 + 3))},
        index=pd.Index(ids, name="player_id"))

    # do the numbers repeat? both steps fitted again, separately, on two halves of the games (folds 0, 2, 4
    # against folds 1, 3), so that nothing from one half leaks into the other
    halves = []
    for ks in ((0, 2, 4), (1, 3)):
        rows_h = np.flatnonzero(np.isin(fold, ks))
        A1_h, b1_h = sum(parts1[k][0] for k in ks), sum(parts1[k][1] for k in ks)
        level_h = np.linalg.solve(A1_h + lam1 * np.eye(2 * P), b1_h)[cols1].sum(axis=1)
        A_h, b_h = ridge_stats(cols, vals, y - level_h, rows_h, m)
        bh = np.linalg.solve(A_h + lam * eye, b_h)
        seen = np.bincount(np.r_[o[rows_h].ravel(), d[rows_h].ravel()], minlength=P)
        halves.append((bh[k4] - bh[k4 + 2], bh[k4 + 1] - bh[k4 + 3], seen))
    del parts1, A1
    both = np.minimum(halves[0][2], halves[1][2])
    regular = both >= np.median(both[both > 0])             # players with a fair amount of play in both halves
    rep_stops = np.corrcoef(halves[0][0][regular], halves[1][0][regular])[0, 1]
    rep_keeps = np.corrcoef(halves[0][1][regular], halves[1][1][regular])[0, 1]
    repeat_text = (f"one half of the games against the other, {regular.sum():,} players with a fair amount of play in "
                   f"both: STOPS {rep_stops:+.2f}, KEEPS {rep_keeps:+.2f} (near 0 = luck)")

    helps = errors[li] < nothing and li < len(LAMBDAS) - 1
    verdict = (f"yes - beyond how good each player is, knowing who is on the floor improves the prediction of games "
               f"left out of the fit (shrinkage chosen: {lam:,.0f})" if helps else
               "no - beyond how good each player is, knowing who is on the floor does not improve the prediction of "
               "games left out of the fit, so the data cannot tell players apart on this")
    return out, verdict, repeat_text, helps


# ============================================================== the three views, for one set of games
def analyse(mask, label, with_overall):
    print(f"\n\n{'#' * 100}\n{label}\n{'#' * 100}")
    saved = []
    ok = mask & has_lineup

    # who was on the floor, for every (season, player)
    ps_key = season.astype("int64") * 100_000_000
    key_off = (ps_key[ok][:, None] + off_players[ok]).ravel()
    key_def = (ps_key[ok][:, None] + def_players[ok]).ravel()
    uniq, inv = np.unique(np.r_[key_off, key_def], return_inverse=True)
    code_off, code_def, K = inv[:len(key_off)], inv[len(key_off):], len(uniq)
    seasons_played = pd.Series(uniq % 100_000_000).value_counts()

    # ---------------------------------------------------------------- VIEW 1
    print("\n" + "=" * 100 + "\nVIEW 1 - ON THE FLOOR (teammates and opponents accounted for)\n" + "=" * 100)
    if with_overall:
        avg = season_average(effects, "net", "net_se")
        for c in ("off", "def"):
            avg[c] = effects.groupby("player_id")[c].mean()
        saved.append(present("STARTS IT - effect on his team's scoring margin whenever he is on the floor", avg,
                             repeats(effects, "net"), extra=("off", "def"), already_fair=True,
                             note="off = his team's offense, def = points the opponent scores (negative is good)"))
    inter, verdict, repeat_text, helps = interaction_model(np.flatnonzero(ok))
    inter["seasons"] = seasons_played.reindex(inter.index).fillna(0).astype(int)
    print(f"\n  Do players differ in how they respond to momentum?  {verdict}.")
    saved.append(present("STOPS IT - extra help when momentum is against his team (per typical swing)",
                         inter.rename(columns={"stops": "value", "stops_se": "se"}), repeat_text,
                         extra=("stop_scoring", "stop_defense"), already_fair=True, nobody=not helps,
                         note="stop_scoring = his team's offense; stop_defense = points kept off the opponent; "
                              "beyond his usual level when he plays; values are shrunk toward 0"))
    saved.append(present("KEEPS IT - extra help when his team already has momentum (per typical swing)",
                         inter.rename(columns={"keeps": "value", "keeps_se": "se"}), repeat_text, already_fair=True, nobody=not helps,
                         note="beyond his usual level when he plays; values are shrunk toward 0"))

    # ---------------------------------------------------------------- VIEW 2
    print("\n" + "=" * 100 + "\nVIEW 2 - CREDIT FOR THE PLAYS (per 100 possessions on the floor)\n" + "=" * 100)
    use_ev = ok[ei]
    # offense: the possession he finished (shot, free throw, turnover) or assisted on
    op = pd.concat([pd.DataFrame({"pid": pid1, "i": ei})[use_ev & np.isin(etype, [1, 2, 3, 5])],
                    pd.DataFrame({"pid": pid2, "i": ei})[use_ev & (etype == 1) & (pid2 > 0)]]).drop_duplicates()
    # defense: the possession he ended with a steal, a block or a defensive rebound
    dp = pd.concat([pd.DataFrame({"pid": pid2, "i": ei})[use_ev & (etype == 5) & text.str.contains("STEAL").to_numpy()],
                    pd.DataFrame({"pid": pid3, "i": ei})[use_ev & (etype == 2) & text.str.contains("BLOCK").to_numpy()],
                    pd.DataFrame({"pid": pid1, "i": ei})[use_ev & (etype == 4) & (p1_side == -sign[ei]) & has_team1]]).drop_duplicates()

    def coded(d_):
        d_ = d_[(d_["pid"] > 0) & (d_["pid"] < 100_000_000)]
        k = ps_key[d_["i"].to_numpy()] + d_["pid"].to_numpy()
        c = np.minimum(np.searchsorted(uniq, k), K - 1)
        found = uniq[c] == k                                  # the player must be among the ten we have on the floor
        return c[found], d_["i"].to_numpy()[found]

    (op_code, op_i), (dp_code, dp_i) = coded(op), coded(dp)
    op_v, dp_v = rate_raw[op_i], -rate_raw[dp_i]              # points better than expected / points SAVED

    def credit(w_off, w_def):
        wo, wd = np.repeat(w_off[ok], 5), np.repeat(w_def[ok], 5)
        f_off, f_def = np.bincount(code_off, wo, K), np.bincount(code_def, wd, K)
        f_sq = np.bincount(code_off, wo * wo, K) + np.bincount(code_def, wd * wd, K)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (np.bincount(op_code, op_v * w_off[op_i], K) / f_off + np.bincount(dp_code, dp_v * w_def[dp_i], K) / f_def)
            n_eff = (f_off + f_def) ** 2 / f_sq               # how many possessions' worth of information
        return value, n_eff

    odd = (pd.factorize(game)[0] % 2).astype(float)           # odd / even games, to measure the luck

    def with_luck(w_off, w_def, minus=None):
        """A credit rate and its uncertainty. How far players' odd-game and even-game numbers sit apart,
        for a given amount of playing time, tells us how much luck there is (pooled over all players)."""
        def rate(h):
            v = credit(w_off * h, w_def * h)[0]
            return v if minus is None else v - credit(minus[0] * h, minus[1] * h)[0]
        full, n_eff = rate(one), credit(w_off, w_def)[1]
        gap = (rate(odd) - rate(1 - odd)) ** 2 * n_eff
        k2 = np.nanmedian(gap[np.isfinite(gap)]) / 4 / 0.455  # median-based, so a few wild values don't matter
        return full, np.sqrt(k2 / n_eff)

    v2 = pd.DataFrame({"season": uniq // 100_000_000, "player_id": uniq % 100_000_000})
    v2["creates"], v2["creates_se"] = with_luck(one, one)
    v2["stops"], v2["stops_se"] = with_luck(against, ahead)
    v2["lift"], v2["lift_se"] = with_luck(against, ahead, minus=(one, one))
    v2 = v2.replace([np.inf, -np.inf], np.nan)
    print("  The first two lists measure involvement (mostly assists, defensive rebounds, steals and blocks), not momentum;")
    print("  only LIFT compares a player with himself.")
    for col, title in (("creates", "CREDIT - points better than expected on the possessions he is credited on (involvement)"),
                       ("stops", "CREDIT WHEN MOMENTUM IS AGAINST - the same, counted more the more momentum was against his team (involvement)"),
                       ("lift", "LIFT (credit) - does he do more than his usual when momentum is against his team?")):
        saved.append(present(title, season_average(v2, col, col + "_se"), repeats(v2, col)))

    # ---------------------------------------------------------------- VIEW 3
    print("\n" + "=" * 100 + f"\nVIEW 3 - WHAT HAPPENS AFTER HIS PLAYS (his team's momentum ahead, half-life {H:.1f})\n" + "=" * 100)
    j = np.minimum(np.arange(n) + 1, n - 1)
    has_next = (np.arange(n) + 1 < n) & (game[j] == game)
    scored = poss["points"].to_numpy() > 0
    made_fg = (poss["fg2m"] + poss["fg3m"]).to_numpy() > 0
    nxt = np.minimum(ei + 1, n - 1)
    is_dunk = (etype == 1) & text.str.contains("Dunk").to_numpy()
    is_block = (etype == 2) & text.str.contains("BLOCK").to_numpy()
    is_and1 = (etype == 3) & text.str.contains("Free Throw 1 of 1").to_numpy() & made_fg[ei]
    is_steal = ((etype == 5) & text.str.contains("STEAL").to_numpy() & (ei + 1 < n) & (game[nxt] == game[ei])
                & (sign[nxt] == -sign[ei]) & scored[nxt])
    first_basket = (etype == 1) & ~ev.duplicated(["i", "etype"]).to_numpy()

    def after(ahead_col):
        ahead_home = np.where(has_next, poss[ahead_col].to_numpy()[j] * sign[j], np.nan)
        base = (pd.DataFrame({"pid": pid1, "i": ei, "x": ahead_home[ei] * p1_side})[use_ev & has_team1]
                .dropna().drop_duplicates(["pid", "i"]).groupby("pid")["x"].mean())           # his everyday baseline

        def score(pid, at, team, w, m_):
            d_ = pd.DataFrame({"player_id": pid[m_], "i": at[m_], "team": team[m_], "w": w[m_]}).drop_duplicates(["player_id", "i"])
            d_ = d_[(d_["player_id"] > 0) & mask[d_["i"].to_numpy()]]
            d_["x"] = ahead_home[d_["i"].to_numpy()] * d_["team"] - d_["player_id"].map(base).fillna(0)
            d_ = d_.dropna(subset=["x"]).assign(season=lambda t: season[t["i"].to_numpy()], g=lambda t: game[t["i"].to_numpy()])
            # Plays in the same game share the possessions that follow them, so they are not independent.
            # Model: a game's average = the player's level + a game-wide part + play-by-play noise.
            # Both noise sizes are estimated once, from all players together, so a player with a handful
            # of plays cannot look precise by accident.
            pg = (d_.assign(xw=d_["x"] * d_["w"], w2=d_["w"] ** 2)
                  .groupby(["player_id", "season", "g"])[["xw", "w", "w2"]].sum().reset_index())
            tot = pg.groupby(["player_id", "season"])[["xw", "w"]].transform("sum")
            level = tot["xw"] / tot["w"]
            share = pg["w"] / tot["w"]                               # how much of his season this game is
            h = pg["w2"] / pg["w"] ** 2                              # 1 / (number of plays in the game)
            fit = share < 0.5
            dev2 = ((pg["xw"] / pg["w"] - level) ** 2 / (1 - share))[fit]
            game_part, play_part = np.maximum(np.linalg.lstsq(
                np.column_stack([np.ones(fit.sum()), h[fit]]), dev2, rcond=None)[0], 0)
            pg["var"] = pg["w"] ** 2 * game_part + pg["w2"] * play_part
            g = pg.groupby(["player_id", "season"])
            return pd.DataFrame({"value": g["xw"].sum() / g["w"].sum(), "se": np.sqrt(g["var"].sum()) / g["w"].sum()}).reset_index()

        creators = pd.concat([score(pid1, ei, sign[ei], one[ei], is_dunk & use_ev), score(pid3, ei, -sign[ei], one[ei], is_block & use_ev),
                              score(pid2, nxt, sign[nxt], one[ei], is_steal & use_ev), score(pid1, ei, sign[ei], one[ei], is_and1 & use_ev)])
        creators = creators.assign(w=1 / creators["se"] ** 2)          # his four kinds of highlight play, combined
        g = creators.assign(vw=creators["value"] * creators["w"]).groupby(["player_id", "season"])
        creators = pd.DataFrame({"value": g["vw"].sum() / g["w"].sum(), "se": 1 / np.sqrt(g["w"].sum())}).reset_index()
        stoppers = score(pid1, ei, sign[ei], against[ei], first_basket & use_ev & (against[ei] > 0))
        return creators, stoppers

    creators, stoppers = after("ahead_lineup")
    saved.append(present("CREATES (after) - after his dunks, blocks, steals-and-scores and and-ones",
                         season_average(creators, "value", "se"), repeats(creators, "value")))
    saved.append(present("STOPS (after) - after his baskets, counted more the more momentum was against his team",
                         season_average(stoppers, "value", "se"), repeats(stoppers, "value")))
    for h, col in ((H_LO, "ahead_lineup_lo"), (H_HI, "ahead_lineup_hi")):
        c_, s_ = after(col)
        print(f"\n  Robustness - with a half-life of {h:g} instead of {H:.1f}:  creators repeat {repeats(c_, 'value')};")
        print(f"      stoppers repeat {repeats(s_, 'value')}")
    return pd.concat(saved).assign(games=label)


out = [analyse(~playoffs, "REGULAR SEASON", with_overall=True)]
if playoffs.any():
    out.append(analyse(playoffs, "PLAYOFFS (each year's playoffs = one season; far fewer games, so far less certain)", with_overall=False))
if DB == "data/nba.db":
    Path("results").mkdir(exist_ok=True)
    Path("results/creators.csv").unlink(missing_ok=True)
    pd.concat(out).to_csv("results/players.csv", index=False, float_format="%.3f")
    print("\nSaved results/players.csv")
