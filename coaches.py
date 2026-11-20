"""
NBA Momentum Study - Step 6: what do a coach's timeouts do to momentum?

Run after momentum.py and download_coaches.py.

The question for every timeout a coach CHOSE to call: compared with not
calling it at that same moment, what happened next to
    - his own team's scoring rate,
    - the other team's scoring rate,
    - and the difference between them (momentum)?
All in points per 100 possessions, looking ahead with the nearest
possessions counting most (the reference half-life set in momentum.py). No "run of
6+", no fixed number of possessions.

Which timeouts count: only those the coach chose. The league forces
stoppages for TV in every quarter (under 7:00 and under 3:00 since 2017-18;
under 6:00 and 3:00, plus 9:00 in the 2nd and 4th quarters, before that)
and the data does not label them, so a timeout that fills one of those
required slots is left out - both as a timeout and as a comparison moment.
Timeouts where the OTHER team gets the ball next are included.

How "compared with not calling it" is done (doubly robust estimation):
  Coaches do not call timeouts at random - they call them when things are
  going badly, and things going badly tend to get better on their own. Two
  models guard against that, each fitted on games the moment is not in:
   1. what normally happens next from this exact situation WITHOUT a timeout
      (momentum of both teams, who has the ball, score, clock, period, home,
      how the last possession ended);
   2. how likely a timeout was in this situation, used to weight the
      no-timeout moments so they resemble the moments when timeouts are
      called.
  The estimate is right if either model is right. Each coach is compared
  with his OWN team's no-timeout moments that season, so a weak roster does
  not count against him.

Per coach: worked out season by season, then averaged over his seasons
(each season counting by how many timeouts it has, i.e. how precisely it is
measured).
    effect on his team   + = his team scores more after his timeouts (GOOD)
    effect on opponent   + = the opponent scores more (BAD)
    impact on momentum   = the first minus the second.  + = GOOD, - = BAD
"+/-" is the 95% margin of error; "fair estimate" pulls each coach toward
the average according to how much of the spread between coaches is luck;
the verdict says whether his impact is clearly positive, clearly negative,
or can't be told from zero. "Repeats?" checks one season against the next.

Usage:
    python coaches.py [--db data/fake_nba.db]
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
FOLDS = 5
KNOT_PCT = [5, 27.5, 50, 72.5, 95]
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)

def would_be_start(con, poss):
    """How the previous possession ended, from its last recorded play: after a made basket, a made free throw, a steal
    (live ball, a fast break is possible), a dead-ball turnover, a defensive rebound by a player (live ball), a team
    rebound (dead ball), or other. That is how the next possession would have started had nobody called a timeout.
    The old covariate (scored / turnover / miss) put live and dead balls together; timeouts come almost only from
    dead balls, so the comparison moments held several times more fast breaks than the timeouts did."""
    ev = pd.read_sql_query("""SELECT game_id, poss_num, seconds_left, eventnum, eventmsgtype AS t, pts, player1_id AS p1, player2_id AS p2
                              FROM events WHERE eventmsgtype IN (1, 2, 3, 4, 5) AND poss_num IS NOT NULL""", con)
    last = ev.sort_values(["game_id", "poss_num", "seconds_left", "eventnum"], ascending=[True, True, False, True]).groupby(
        ["game_id", "poss_num"], sort=False).tail(1)
    team_id = lambda v: (v >= 1610612737) & (v <= 1610612766)     # team rebounds and team turnovers carry the team's id
    p1, p2 = last["p1"].fillna(0).to_numpy(), last["p2"].fillna(0).to_numpy()
    t, pts = last["t"].to_numpy(), last["pts"].fillna(0).to_numpy()
    cls = np.select([t == 1, (t == 3) & (pts > 0), (t == 5) & (p2 > 0) & ~team_id(p2), t == 5, (t == 4) & (p1 > 0) & ~team_id(p1), t == 4],
                    [0, 1, 2, 3, 4, 5], 6)
    end_cls = pd.Series(cls, index=pd.MultiIndex.from_arrays([last["game_id"], last["poss_num"]])).reindex(
        pd.MultiIndex.from_arrays([poss["game_id"], poss["poss_num"]])).fillna(6).astype(int).to_numpy()
    return np.r_[6, end_cls[:-1]]                       # the possession before (never used for the first of a period)


# ============================================================== load data
con = sqlite3.connect(DB)
# make the database use the real (game, possession) indexes when joining tables; left to itself it
# sometimes builds a makeshift index on half the key, which makes loading take many minutes
con.execute("PRAGMA automatic_index = OFF")
try:
    poss = pd.read_sql_query(
        """SELECT p.game_id, p.poss_num, p.season, p.playoffs, p.period, p.seconds_left, p.is_home, p.home, p.away,
                  p.starttype, p.points, p.turnovers, p.start_margin,
                  m.rate_raw, m.rate_lineup, m.before_off, m.before_def, m.ahead_own, m.ahead_opp, m.ahead_lineup, p.end_seconds_left
           FROM poss p JOIN poss_momentum m USING (game_id, poss_num) ORDER BY p.game_id, p.poss_num""", con)
    H = float(pd.read_sql_query("SELECT value FROM momentum_settings WHERE name = 'half_life'", con)["value"].iloc[0])
    coach_of = pd.read_sql_query("SELECT * FROM game_coach", con).set_index(["game_id", "team"])["coach"]
except Exception as e:
    raise SystemExit(f"Run momentum.py and download_coaches.py first. [{e}]")
timeouts = pd.read_sql_query(
    """SELECT game_id, period, seconds_left, homedescription AS home_txt, visitordescription AS away_txt
       FROM events WHERE eventmsgtype = 9""", con)
start_class = would_be_start(con, poss)
con.close()

n = len(poss)
pos = np.arange(n)
game = poss["game_id"].to_numpy()
gidx = pd.factorize(game)[0]
period = poss["period"].to_numpy()
clock = poss["seconds_left"].to_numpy()
season = poss["season"].to_numpy()
playoffs = (poss["playoffs"] == 1).to_numpy()
home_ball = (poss["is_home"] == 1).to_numpy()
first_of_period = np.r_[True, (gidx[1:] != gidx[:-1]) | (period[1:] != period[:-1])]

# ------------------------------------------------------- timeouts: who called it, and was it a forced TV slot?
timeouts = timeouts[timeouts["game_id"].isin(np.unique(game))].copy()
timeouts["season"] = timeouts["game_id"].map(pd.Series(season, index=game).groupby(level=0).first())
timeouts = timeouts.sort_values(["game_id", "period", "seconds_left"], ascending=[True, True, False])
taken = timeouts.groupby(["game_id", "period"]).cumcount().to_numpy()         # timeouts already taken this quarter
t_clock, t_period = timeouts["seconds_left"].to_numpy(), timeouts["period"].to_numpy()
required = np.where(timeouts["season"].to_numpy() >= 2017, (t_clock <= 419) * 1 + (t_clock <= 179),
                    (t_clock <= 359) * 1 + (t_clock <= 179) + ((t_clock <= 539) & np.isin(t_period, [2, 4])))
forced = (taken < required) & (t_period <= 4)
game_no = pd.Series(np.arange(gidx.max() + 1), index=pd.unique(game))
key = gidx * 10_000 + period * 1_000 + (720 - clock)
t_key = game_no.reindex(timeouts["game_id"]).to_numpy() * 10_000 + t_period * 1_000 + (720 - t_clock)
tj = np.minimum(np.searchsorted(key, t_key, side="left"), n - 1)             # the possession that follows it
good = (game[tj] == timeouts["game_id"].to_numpy()) & (period[tj] == t_period)
after_timeout = poss["starttype"].fillna("").to_numpy() == "Off Timeout"
# A timeout whose clock falls inside the window of an "Off Timeout" possession belongs to that possession (the
# possession is dated from the stoppage before it). Skipping these used to drop about half of the timeouts.
j0 = np.maximum(tj - 1, 0)
inside = ((game[j0] == timeouts["game_id"].to_numpy()) & (period[j0] == t_period) & (tj > 0) & (clock[j0] > t_clock)
          & (t_clock >= poss["end_seconds_left"].to_numpy()[j0]))
use_prev = after_timeout[j0] & inside & ~(good & (clock[tj] == t_clock) & after_timeout[tj])
tj, good = np.where(use_prev, j0, tj), good | use_prev
called_by = np.zeros(n)                                                       # +1 home / -1 away
is_forced = np.zeros(n, bool)
# who called it: the team named in its own description column. "Timeout: Official" (2015-17) and a few "Timeout:
# Regular" lines name no team - an official's stoppage, nobody's choice - so they are left out like the TV slots.
team_named = (timeouts["home_txt"].notna() | timeouts["away_txt"].notna()).to_numpy()
called_by[tj[good]] = np.where(timeouts["home_txt"].notna(), 1, np.where(timeouts["away_txt"].notna(), -1, 0))[good]
np.logical_or.at(is_forced, tj[good], (forced | ~team_named)[good])
print(f"Reference half-life from momentum.py: {H:.1f}.  Unit: points per 100 possessions.")
print(f"Timeouts in the data: {good.sum():,}; left out as forced TV slots: {forced[good].sum():,} "
      f"({forced[good].mean():.0%}); official, no team named: {(~team_named & ~forced)[good].sum():,}; "
      f"coach-chosen: {(team_named & ~forced)[good].sum():,}")
print(f"Each timeout is matched to the possession it leads into ('Off Timeout'); 'Off Timeout' possessions left without a "
      f"caller: {(after_timeout & (called_by == 0)).sum():,}.\n")

# ------------------------------------------------------- one row per (stoppage, team): its view of that moment
# view 0 = the team that gets the ball next, view 1 = the team that will defend
prev_end = start_class                                  # how the possession would have started (see would_be_start)
usable = ~first_of_period & ~(after_timeout & (called_by == 0)) & ~is_forced
rows = np.flatnonzero(usable)
R = len(rows)
idx = np.r_[rows, rows]
has_ball = np.r_[np.ones(R, bool), np.zeros(R, bool)]
side = np.where(has_ball, np.where(home_ball[idx], 1, -1), np.where(home_ball[idx], -1, 1))   # +1 home / -1 away
flip = np.where(has_ball, 1.0, -1.0)
mom = (poss["before_off"] - poss["before_def"]).to_numpy()[idx] * flip        # this team's momentum before
SD = mom.std()
treated = after_timeout[idx] & (called_by[idx] == side)
other_called = after_timeout[idx] & (called_by[idx] == -side)
keep = ~other_called                                                          # the opponent's timeouts are not our comparison
own_ahead = np.where(has_ball, poss["ahead_own"].to_numpy()[idx], poss["ahead_opp"].to_numpy()[idx])
opp_ahead = np.where(has_ball, poss["ahead_opp"].to_numpy()[idx], poss["ahead_own"].to_numpy()[idx])
own_before = np.where(has_ball, poss["before_off"].to_numpy()[idx], poss["before_def"].to_numpy()[idx])
opp_before = np.where(has_ball, poss["before_def"].to_numpy()[idx], poss["before_off"].to_numpy()[idx])
OUTCOMES = {"his team's scoring rate": own_ahead, "the opponent's scoring rate": opp_ahead,
            "momentum (his team minus opponent)": own_ahead - opp_ahead,
            "momentum, chain and players removed": poss["ahead_lineup"].to_numpy()[idx] * flip}


def spline(x, source=None):
    k = np.unique(np.percentile(x if source is None else source, KNOT_PCT))
    if len(k) < 3:
        return x[:, None]
    cube = lambda v: np.maximum(v, 0) ** 3
    cols = [x] + [(cube(x - k[j]) - cube(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                   + cube(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2 for j in range(len(k) - 2)]
    return np.column_stack(cols)


margin = poss["start_margin"].to_numpy(float)[idx] * flip
late = (period[idx] >= 4).astype(float)
X = np.column_stack([
    np.ones(2 * R), has_ball, (side == 1), spline(mom / SD), (mom / SD) * has_ball, own_before / SD, opp_before / SD,
    spline(margin), spline(margin) * late[:, None], late, spline(np.log1p(clock[idx])),
    pd.get_dummies(np.minimum(period[idx], 5)).to_numpy(float)[:, 1:],
    pd.get_dummies(prev_end[idx]).to_numpy(float)[:, 1:], pd.get_dummies(prev_end[idx]).to_numpy(float)[:, 1:] * has_ball[:, None],
    playoffs[idx]]).astype(float)
X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / np.where(X[:, 1:].std(0) > 0, X[:, 1:].std(0), 1)   # same scale, for stable fitting
fold = gidx[idx] % FOLDS
g_row = gidx[idx]

# ------------------------------------------------------- the two models, each fitted without the moment's own games
propensity = np.zeros(2 * R)
expected0 = {name: np.zeros(2 * R) for name in OUTCOMES}
for f in range(FOLDS):
    fit, out = keep & (fold != f), fold == f
    # 2. chance of a timeout (logistic regression, fitted by Newton steps)
    beta = np.zeros(X.shape[1])
    Xf, tf = X[fit], treated[fit].astype(float)
    for _ in range(25):
        p_ = 1 / (1 + np.exp(-np.clip(Xf @ beta, -30, 30)))
        step = np.linalg.solve((Xf * (p_ * (1 - p_))[:, None]).T @ Xf + 1e-6 * np.eye(X.shape[1]), Xf.T @ (tf - p_))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    propensity[out] = 1 / (1 + np.exp(-np.clip(X[out] @ beta, -30, 30)))
    # 1. what happens next without a timeout (linear regression on the no-timeout moments)
    c = fit & ~treated
    inv = np.linalg.pinv(X[c].T @ X[c])
    for name, y in OUTCOMES.items():
        expected0[name][out] = X[out] @ (inv @ (X[c].T @ y[c]))
weight = np.where(treated, 0.0, np.minimum(propensity / (1 - propensity), 50.0)) * keep   # controls weighted to look like timeouts
resid = {name: OUTCOMES[name] - expected0[name] for name in OUTCOMES}


def components(r, rows, w=None):
    """Pooled noise model: a game's average = the level + a game-wide part + possession-by-possession
    noise. Returns the size of the two noise parts, estimated from all the rows given. Using this pooled
    model (not each coach's own handful of games) keeps small samples from looking precise by accident."""
    w = np.ones(rows.sum()) if w is None else w
    d = pd.DataFrame({"g": g_row[rows], "xw": r[rows] * w, "w": w, "w2": w * w}).groupby("g").sum()
    d = d[d["w"] > 0]
    dev2 = (d["xw"] / d["w"] - d["xw"].sum() / d["w"].sum()) ** 2
    return np.maximum(np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["w2"] / d["w"] ** 2]), dev2, rcond=None)[0], 0)


def effect(mask_t, mask_c, name, noise=None):
    """Doubly robust effect on the timeouts in mask_t, using the comparison moments in mask_c.
    `name` is one of the outcomes, or an array of raw values (then it is a weighted comparison only,
    with missing values skipped). `noise` = pooled noise sizes for (timeouts, comparison moments);
    if not given they are estimated from the rows themselves."""
    r = resid[name] if isinstance(name, str) else name
    seen = ~np.isnan(r)
    mask_t, mask_c = mask_t & seen, mask_c & seen
    game_part, play_part = noise[0] if noise else components(r, mask_t)
    t = pd.DataFrame({"g": g_row[mask_t], "x": r[mask_t], "n": 1.0}).groupby("g").sum()
    est_t = t["x"].sum() / t["n"].sum()
    var_t = (game_part * (t["n"] ** 2).sum() + play_part * t["n"].sum()) / t["n"].sum() ** 2
    w = weight[mask_c]
    if w.sum() == 0:
        return est_t, np.sqrt(var_t)
    game_part, play_part = noise[1] if noise else components(r, mask_c, w)
    c = pd.DataFrame({"g": g_row[mask_c], "x": r[mask_c] * w, "n": w, "n2": w * w}).groupby("g").sum()
    est_c = c["x"].sum() / c["n"].sum()
    var_c = (game_part * (c["n"] ** 2).sum() + play_part * c["n2"].sum()) / c["n"].sum() ** 2
    return est_t - est_c, np.sqrt(var_t + var_c)


def fair(value, se):
    w = 1 / se ** 2
    centre = np.sum(w * value) / np.sum(w)
    real_var = max(np.sum(w * (value - centre) ** 2) / np.sum(w) - np.sum(w * se ** 2) / np.sum(w), 0.0)
    return centre + real_var / (real_var + se ** 2) * (value - centre), np.sqrt(real_var)


# each team's own possessions in order, to look up "its k-th trip from this stoppage on"
team_game = gidx * 2 + home_ball
order = np.argsort(team_game, kind="stable")
slot = pos - np.maximum.accumulate(np.where(np.r_[True, gidx[1:] != gidx[:-1]], pos, 0))
seq_key, seq_tg = team_game[order] * 1_000_000 + slot[order], team_game[order]
seq_rate = poss["rate_raw"].to_numpy()[order]
own_tg, opp_tg = gidx[idx] * 2 + (side == 1), gidx[idx] * 2 + (side == -1)
own_at = np.searchsorted(seq_key, own_tg * 1_000_000 + slot[idx])
opp_at = np.searchsorted(seq_key, opp_tg * 1_000_000 + slot[idx])


def trip(at, tg, k):
    """Scoring rate (vs expectation) on a team's k-th possession from the stoppage on."""
    a = np.minimum(at + k, n - 1)
    return np.where((at + k < n) & (seq_tg[a] == tg), seq_rate[a], np.nan)


TRIPS = 6
own_trip = [trip(own_at, own_tg, k) for k in range(TRIPS)]
opp_trip = [trip(opp_at, opp_tg, k) for k in range(TRIPS)]

team_name = np.where(side == 1, poss["home"].to_numpy()[idx], poss["away"].to_numpy()[idx])
coach = coach_of.reindex(pd.MultiIndex.from_arrays([game[idx], team_name])).to_numpy()
coach_season = pd.Series(coach, dtype="object").fillna("?").to_numpy() + "|" + season[idx].astype(str)
cs_code, cs_names = pd.factorize(coach_season)              # numbers compare much faster than text
control = keep & ~treated


def report(scope, label):
    t_all, c_all = treated & scope, control & scope
    print(f"\n{'#' * 100}\n{label}: {t_all.sum():,} coach-chosen timeouts, {c_all.sum() // 2:,} comparison stoppages\n{'#' * 100}")
    if t_all.sum() < 200:
        print("  Too few timeouts to say anything.")
        return None

    # ---------------------------------------------------------------- 1. league-wide
    print("\n1. LEAGUE-WIDE: what a coach-chosen timeout changes, compared with not calling it at that moment")
    rows_ = []
    for name in OUTCOMES:
        est, se = effect(t_all, c_all, name)
        row = {"": name, "all timeouts": f"{est:+.2f} (+/-{1.96 * se:.2f})"}
        for tag_, m_ in (("his team has the ball next", has_ball), ("the opponent has the ball next", ~has_ball)):
            e2, s2 = effect(t_all & m_, c_all & m_, name)
            row[tag_] = f"{e2:+.2f} (+/-{1.96 * s2:.2f})"
        rows_.append(row)
    league = pd.DataFrame(rows_)
    print(league.to_string(index=False))
    print("   + on his team's rate is good; + on the opponent's rate is bad; + on momentum is good.")

    # does it matter how much momentum was against his team? (straight-line fit through the timeouts)
    base = effect(t_all, c_all, "momentum (his team minus opponent)")[0]
    y_ = resid["momentum (his team minus opponent)"][t_all]
    x_ = np.column_stack([np.ones(t_all.sum()), -mom[t_all] / SD])
    inv = np.linalg.pinv(x_.T @ x_)
    b_ = inv @ (x_.T @ y_)
    sc = pd.DataFrame(x_ * (y_ - x_ @ b_)[:, None]).groupby(g_row[t_all]).sum().to_numpy()
    se_b = np.sqrt(np.diag(inv @ (sc.T @ sc) @ inv))
    print(f"\n   Does a timeout help more when more momentum is against the team?  For each extra typical swing of momentum")
    print(f"   against, the timeout's impact changes by {b_[1]:+.2f} (+/-{1.96 * se_b[1]:.2f}).  "
          f"(Average timeout: momentum {mom[t_all].mean():+.0f} per 100; overall impact {base:+.2f}.)")

    # both teams, before and after, next to what would have happened anyway
    print("\n   Both teams' scoring rates around the average coach-chosen timeout:")
    both = pd.DataFrame({
        "": ["his team", "the opponent"],
        "before the timeout": [own_before[t_all].mean(), opp_before[t_all].mean()],
        "after it": [own_ahead[t_all].mean(), opp_ahead[t_all].mean()],
        "expected after, with no timeout": [own_ahead[t_all].mean() - effect(t_all, c_all, "his team's scoring rate")[0],
                                            opp_ahead[t_all].mean() - effect(t_all, c_all, "the opponent's scoring rate")[0]]})
    print(both.to_string(index=False, float_format="{:+.1f}".format))
    print("   (rates are vs expectation; both teams drift back toward 0 with or without a timeout)")

    # how long does it last? each team's next trips one at a time (weighted comparison with no-timeout moments)
    print("\n   How long does it last?  The timeout's effect on each team's rate, one trip at a time:")
    profile = pd.DataFrame(
        {f"trip {k + 1}": ["{:+.1f} (+/-{:.1f})".format(*(lambda e: (e[0], 1.96 * e[1]))(effect(t_all, c_all, own_trip[k]))),
                           "{:+.1f} (+/-{:.1f})".format(*(lambda e: (e[0], 1.96 * e[1]))(effect(t_all, c_all, opp_trip[k])))]
         for k in range(TRIPS)}, index=["his team", "the opponent"])
    print(profile.to_string())

    # ---------------------------------------------------------------- 2. coaches
    recs = []
    SHORT = (("own", "his team's scoring rate"), ("opp", "the opponent's scoring rate"),
             ("net", "momentum (his team minus opponent)"), ("adj", "momentum, chain and players removed"))
    noise = {name: (components(resid[name], t_all), components(resid[name], c_all, weight[c_all])) for _, name in SHORT}
    for code in np.unique(cs_code[t_all & pd.notna(coach)]):
        k = cs_names[code]
        mk = cs_code == code
        mt, mc = t_all & mk, c_all & mk
        if mt.sum() < 2 or mc.sum() < 2:
            continue
        rec = {"coach": k.split("|")[0], "season": int(k.split("|")[1]), "timeouts": int(mt.sum()), "when": mom[mt].mean()}
        for short, name in SHORT:
            rec[short], rec[short + "_se"] = effect(mt, mc, name, noise[name])
        recs.append(rec)
    cs = pd.DataFrame(recs)
    pairs = cs.merge(cs.assign(season=cs["season"] - 1), on=["coach", "season"], suffixes=("", "_next"))
    print(f"\n2. COACHES.  Repeats?  A coach's number in one season vs his next season ({len(pairs)} pairs of seasons)")
    if len(pairs) >= 20:
        for col, lab in (("when", "how much momentum is against him when he calls one"), ("own", "effect on his team's scoring"),
                         ("opp", "effect on the opponent's scoring"), ("net", "impact on momentum"),
                         ("adj", "impact on momentum, chain and players removed")):
            a_, b2 = pairs[col].to_numpy(), pairs[f"{col}_next"].to_numpy()
            r = np.corrcoef(a_, b2)[0, 1]
            luck = np.array([np.corrcoef(a_, rng.permutation(b2))[0, 1] for _ in range(5000)])
            verdict = "a lasting trait" if r > np.quantile(luck, 0.99) else "weak hint" if r > np.quantile(luck, 0.95) else "no - looks like luck"
            print(f"   {lab:52s} {r:+.2f}  ({verdict})")
    else:
        print("   not enough seasons to say")

    # average of his seasons, each season counting by how precisely it is measured (how many timeouts)
    g = cs.groupby("coach")
    board = pd.DataFrame({"seasons": g.size(), "timeouts": g["timeouts"].sum(), "when": g["when"].mean()})
    for c in ("own", "opp", "net", "adj"):
        w = 1 / cs[c + "_se"] ** 2
        gw = cs.assign(w=w, vw=cs[c] * w).groupby("coach")
        board[c], board[c + "_pm"] = gw["vw"].sum() / gw["w"].sum(), 1.96 / np.sqrt(gw["w"].sum())
    board["fair"], real_sd = fair(board["net"].to_numpy(), board["net_pm"].to_numpy() / 1.96)
    board["verdict"] = np.select([board["net"] - board["net_pm"] > 0, board["net"] + board["net_pm"] < 0],
                                 ["clearly positive", "clearly negative"], "can't tell from zero")
    board["evidence"] = board["net"] / board["net_pm"]          # how strong the evidence is, for ordering ties
    board = board.sort_values(["fair", "evidence"], ascending=False)
    flagged = (board["verdict"] != "can't tell from zero").sum()
    print(f"\n   {len(board)} coaches. Each number is the average of his seasons. Ranked by the fair estimate")
    print("   (when the fair estimates are all equal - no real spread - by how strong the evidence is).")
    print(f"   Real spread between coaches: {real_sd:.2f}.  Clearly positive or negative: {flagged} "
          f"(luck alone would flag about {0.05 * len(board):.0f}).\n")
    show = board.reset_index()[["coach", "seasons", "timeouts", "when", "own", "own_pm", "opp", "opp_pm", "net", "net_pm", "fair", "verdict", "adj"]]
    show.columns = ["coach", "seasons", "timeouts", "momentum when he calls it", "effect on his team", "+/-", "effect on opponent", "+/- ",
                    "impact on momentum", "+/-  ", "fair estimate", "verdict", "impact, chain+players removed"]
    signed, plain = "{:+.1f}".format, "{:.1f}".format
    print(show.to_string(index=False, formatters={
        "timeouts": "{:,}".format, "momentum when he calls it": signed, "effect on his team": signed, "+/-": plain,
        "effect on opponent": signed, "+/- ": plain, "impact on momentum": signed, "+/-  ": plain, "fair estimate": signed,
        "impact, chain+players removed": signed}))
    return show.assign(games=label)


out = [report(~playoffs[idx], "REGULAR SEASON")]
if playoffs.any():
    out.append(report(playoffs[idx], "PLAYOFFS"))
if DB == "data/nba.db":
    Path("results").mkdir(exist_ok=True)
    for old in ("timeouts_league.csv", "timeouts_by_momentum.csv"):
        Path("results", old).unlink(missing_ok=True)
    pd.concat([o for o in out if o is not None]).to_csv("results/coaches.csv", index=False, float_format="%.3f")
    print("\nSaved results/coaches.csv")
