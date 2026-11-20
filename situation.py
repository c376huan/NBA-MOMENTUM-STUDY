"""
NBA Momentum Study - the game situation at the start of every possession, as regression columns.

Shared by expected.py (expected points) and impact.py (player impact), so that both describe the
situation in exactly the same way. expected.py's docstring explains why each part is there: smooth
curves for the score margin and the clock, and the game states the curves miss (end of a period,
end of the game, the foul penalty, days of rest).
"""

import numpy as np
import pandas as pd

KNOT_PCT = [5, 27.5, 50, 72.5, 95]              # standard 5-knot placement for restricted cubic splines


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


def in_penalty(con, poss, clock):
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


def rest_days(poss):
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


def situation_matrix(con, poss):
    """One row per possession of `poss`: intercept, home court, period, smooth curves for the score margin (with a
    separate curve for the 4th quarter and overtime) and the clock, and the game states. Also returns, for the
    printout, the share of possessions with the defense in the foul penalty and the share in the last six minutes
    of the 4th quarter or overtime."""
    n = len(poss)
    margin = poss["start_margin"].to_numpy(float)                 # offense's lead at the start of the possession
    clock = poss["seconds_left"].to_numpy(float)
    period = poss["period"].to_numpy()
    log_clock = np.log1p(clock)
    late = period >= 4
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
        end_game[:, None] == np.arange(36), end_quarter[:, None] == np.arange(3), in_penalty(con, poss, clock),
        rest_days(poss)]).astype(float)
    return situation, situation[:, -7].mean(), (end_game >= 0).mean()
