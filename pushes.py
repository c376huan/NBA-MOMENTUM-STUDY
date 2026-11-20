"""
NBA Momentum Study - Step 7: the big pushes. Who starts them, who leads them,
who stops them, and what timeouts do while one is under way.

Run after momentum.py (and download_coaches.py for the coach table).

WHAT A PUSH IS (no hand-picked cut-off such as "8-0" or "10 possessions")
  Every possession has a result compared with expectation (points above or
  below what the two teams, the score and the clock predict). For a stretch
  of play, add those up for one team against the other and divide by how
  much luck alone moves that sum over a stretch of that length:

      push strength = (points gained vs expectation) / (noise x sqrt(length))

  A team's PUSH in a game is the stretch where that number is highest - big
  and fast. The stretch picks its own start, end and length. This is a
  "scan statistic": it finds the most unusual stretch without being told how
  long to look. Shuffled games (same teams, same night, results in random
  order) show how strong the biggest push gets by chance alone.

  The same number can be worked out LIVE: at any moment, the strength of the
  push in progress is the best stretch that ends right now. That is what the
  timeout part uses, because a coach can only react to what has happened.

COMPARED WITH WHAT? (the team's normal level, or how this game has been going?)
  "Expectation" is the team's NORMAL level: what the two teams, the score and
  the clock predict, learned from the whole season. One could instead compare
  a stretch with how the team has played so far TONIGHT. The fair mix of the
  two is a weighted average with weight m / (m + k) on tonight, where m is
  the number of possessions seen tonight and k is how many possessions of
  evidence the normal level is worth:

      k = (luck in one possession)^2 / (night-to-night spread of a team's true level)^2

  The script measures that night-to-night spread itself (section A): if a
  team really were hotter or colder than normal for a whole game, two
  possessions from DIFFERENT quarters of that game would tend to fall on the
  same side of expectation. In the NBA data they do not - the spread is
  about zero, so the weight on tonight stays near 0% even after a full game
  and pushes are measured against the normal level. At the opening tip m = 0,
  so the comparison is the normal level anyway; no previous game is needed.

PLAYERS (each team's biggest push in each game; bigger pushes count more)
  STARTS   he was credited on the first possession of the push
  LEADS    he had the most credit within it: points better than expected on
           the possessions he finished or assisted, plus points saved on the
           defensive possessions he ended
  STOPS    he was credited on the possession right after the OTHER team's
           push (the one that ended it)
  "Credited" = any play recorded for him on that possession: a shot (made or
  missed), free throws, a turnover, an assist, a steal, a block or a defensive
  rebound. These three lists measure INVOLVEMENT. A push's first possession is
  good for the pushing team by construction and the one after it is bad, so
  the players credited most often are simply those involved in the most plays
  all the time: a player's share of his team's credited plays explains the
  differences between players in these lists completely. Whether anyone does
  MORE than his usual during pushes is the next section.
  Each is a rate: out of the pushes that happened while he was on the floor.
  Worked out season by season, averaged (each season counting by how much
  information it holds), pulled toward the league average according to how
  much is luck, and tagged clearly above / clearly below / can't tell.

MORE THAN HIS USUAL (the same plays, measured against the player himself)
  The three lists above reward being involved, so a player who makes most of
  his team's plays ALL the time leads them. This part asks the sharper
  question: during a push, does he do more than he normally does?
  Every credited play is given a chance of having been his: how often he is
  usually credited on that kind of possession (a possession that scored 0,
  1, 2 or 3 points; or, for a defender, a stop), shared out among the five
  on the floor at that moment. Over his whole season those chances add up
  exactly to what he actually did, so "his usual" really is his usual.
    STARTS   how often he made the first play of a push, minus how often his
             usual share says he would
    CARRIES  his share of all the credit in his team's pushes, minus his
             usual share (the counterpart of LEADS)
    STOPS    how often he made the play that ended the other team's push,
             minus his usual
  Positive = he does more than his usual when it matters; zero = the same as
  always (a star who dominates all the time scores zero here); negative =
  less. Luck: each play is a coin with his usual chance p, and the luck in a
  sum of coins is the sum of p(1-p), scaled up by how much plays in the same
  push move together (measured from the data).

TIMEOUTS (coach-chosen only, as in coaches.py)
  At every stoppage we know how strong the opponent's push in progress is.
  Compared with not calling a timeout at the same moment (same push
  strength, score, clock, who has the ball...), how much FURTHER does the
  push go, and how do the next possessions go? Shown overall and by how
  strong the push was. Then coach by coach, timeouts during stronger pushes
  counting more.

BENCH PLAYERS
  The same three lists for players who came off the bench: in each game the
  five on the floor for a team's opening possession are its starters, and
  everyone else is bench for that game. Only bench games are counted, and
  bench players are compared with other bench players.

Usage:
    python pushes.py [--db data/fake_nba.db] [--shuffles 5] [--top 12]
    python pushes.py --shuffles 0 --skip-timeouts     # player lists only (faster)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


DB = arg("--db", "data/nba.db")
B = arg("--shuffles", 5)
TOP = arg("--top", 12)
SKIP_TIMEOUTS = "--skip-timeouts" in sys.argv        # players only (much faster)
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
        """SELECT p.game_id, p.poss_num, p.season, p.playoffs, p.gamedate, p.period, p.seconds_left, p.is_home, p.home, p.away,
                  p.starttype, p.points, p.def_points, p.turnovers, p.start_margin, p.end_seconds_left,
                  l.o1, l.o2, l.o3, l.o4, l.o5, l.d1, l.d2, l.d3, l.d4, l.d5, l.complete,
                  m.rate_raw, m.before_off, m.before_def, m.ahead_own, m.ahead_opp
           FROM poss p JOIN poss_momentum m USING (game_id, poss_num) JOIN lineups l USING (game_id, poss_num)
           ORDER BY p.game_id, p.poss_num""", con)
    names = pd.read_sql_query("SELECT * FROM players", con).drop_duplicates("player_id").set_index("player_id")["player"]
except Exception as e:
    raise SystemExit(f"Run build_lineups.py, expected.py and momentum.py first. [{e}]")
try:
    coach_of = pd.read_sql_query("SELECT * FROM game_coach", con).set_index(["game_id", "team"])["coach"]
except Exception:
    coach_of = None
ev = pd.read_sql_query(
    """SELECT game_id, poss_num, eventmsgtype AS etype, homedescription AS home_txt, visitordescription AS away_txt,
              player1_id AS p1, player1_team_abbreviation AS t1, player2_id AS p2, player3_id AS p3
       FROM events WHERE poss_num IS NOT NULL AND eventmsgtype <= 5""", con)
timeouts = pd.read_sql_query(
    """SELECT game_id, period, seconds_left, homedescription AS home_txt, visitordescription AS away_txt
       FROM events WHERE eventmsgtype = 9""", con)
start_class = would_be_start(con, poss) if not SKIP_TIMEOUTS else None
con.close()

n = len(poss)
pos = np.arange(n)
game = poss["game_id"].to_numpy()
gidx = pd.factorize(game)[0]
G = gidx.max() + 1
season = poss["season"].to_numpy()
playoffs = (poss["playoffs"] == 1).to_numpy()
period = poss["period"].to_numpy()
clock = poss["seconds_left"].to_numpy()
home_ball = (poss["is_home"] == 1).to_numpy()
sign = np.where(home_ball, 1, -1)
x = poss["rate_raw"].to_numpy() / 100 * sign               # HOME team's points gained vs expectation, per possession
SIGMA = x.std()                                             # how much luck moves one possession
home_pts = np.where(home_ball, poss["points"], poss["def_points"]).astype(float)
away_pts = np.where(home_ball, poss["def_points"], poss["points"]).astype(float)
elapsed = np.where(period <= 4, (period - 1) * 720 + (720 - clock), 2880 + (period - 5) * 300 + (300 - clock))
game_start = np.flatnonzero(np.r_[True, gidx[1:] != gidx[:-1]])
game_len = np.diff(np.r_[game_start, n])
first_of_period = np.r_[True, (gidx[1:] != gidx[:-1]) | (period[1:] != period[:-1])]

# Is there a "level tonight"? Take two possessions from different quarters of the same game and multiply
# their results (each measured against expectation). If teams had good and bad nights beyond their normal
# level, those products would average above zero; that average is the night-to-night variance of the level.
# (Different quarters, so that one possession setting up the next does not count as a "good night".)
by_quarter = pd.DataFrame({"g": gidx, "q": period, "x": x}).groupby(["g", "q"])["x"].agg(["sum", "size"])
in_game = by_quarter.groupby("g").sum()
pair_sum = (in_game["sum"] ** 2 - (by_quarter["sum"] ** 2).groupby("g").sum()).to_numpy()      # sum of x_i * x_j over such pairs
pair_n = (in_game["size"] ** 2 - (by_quarter["size"] ** 2).groupby("g").sum()).to_numpy()      # number of such pairs
TONIGHT_VAR = pair_sum.sum() / pair_n.sum()
boot_rng = np.random.default_rng(1)                         # its own generator, so the shuffles below are unchanged
draws = (boot_rng.integers(0, G, G) for _ in range(1000))   # resample whole games to see how sure the estimate is
TONIGHT_VAR_HI = np.quantile([pair_sum[d].sum() / pair_n[d].sum() for d in draws], 0.975)
halves = pd.DataFrame({"g": gidx, "h": np.where(period <= 2, 1, 2), "x": x})[period <= 4].groupby(["g", "h"])["x"].mean().unstack()

# ============================================================== A. find the pushes
LMAX = game_len.max() + 1
k_ = np.arange(LMAX)
length = k_[None, :] - k_[:, None]                          # length[a, b] = number of possessions in the stretch a..b-1
VALID = length > 0
INV = np.where(VALID, 1 / (SIGMA * np.sqrt(np.maximum(length, 1))), 0.0)


def scan(xg, live=True):
    """Strength of every stretch of a game at once. Returns each side's best stretch (start, end, strength)
    and, if live, the strength and start of the best stretch ending before each possession."""
    L = len(xg)
    c = np.r_[0.0, np.cumsum(xg)]
    z = (c[None, :] - c[:, None]) * INV[:L + 1, :L + 1]
    ok = VALID[:L + 1, :L + 1]
    out = []
    for s in (1, -1):
        zs = np.where(ok, s * z, -np.inf)
        a, b = divmod(int(zs.argmax()), L + 1)
        best = (a, b, zs[a, b])
        if live:
            out.append((best, np.maximum(zs.max(axis=0)[:L], 0), zs.argmax(axis=0)[:L]))
        else:
            out.append(best[2])
    return out


rows = []
live_strength = {1: np.zeros(n), -1: np.zeros(n)}           # strength of each side's push in progress, before each possession
live_start = {1: pos.copy(), -1: pos.copy()}                # ... and the row where that push began
for g0, L in zip(game_start, game_len):
    for s, (best, strength, start) in zip((1, -1), scan(x[g0:g0 + L])):
        a, b, z = best
        rows.append((gidx[g0], s, g0 + a, g0 + b - 1, z))
        live_strength[s][g0:g0 + L] = strength
        live_start[s][g0:g0 + L] = g0 + start
pushes = pd.DataFrame(rows, columns=["g", "side", "start", "end", "strength"])
cx, chp, cap = (np.r_[0.0, np.cumsum(v)] for v in (x, home_pts, away_pts))
st, en = pushes["start"].to_numpy(), pushes["end"].to_numpy()
pushes["gain"] = (cx[en + 1] - cx[st]) * pushes["side"]                    # points gained vs expectation
pushes["n_poss"] = en - st + 1
h_, a_ = chp[en + 1] - chp[st], cap[en + 1] - cap[st]
pushes["pts_for"], pushes["pts_against"] = np.where(pushes["side"] == 1, h_, a_), np.where(pushes["side"] == 1, a_, h_)
pushes["seconds"] = np.maximum(elapsed[np.minimum(en + 1, n - 1)] * (gidx[np.minimum(en + 1, n - 1)] == gidx[en]) - elapsed[st], 0)
pushes["season"], pushes["playoffs"] = season[st], playoffs[st]

# chance: the same search on games whose results are shuffled within each team's quarter
grp = (np.cumsum(first_of_period) * 2 + home_ball).astype(float)
order = np.argsort(grp, kind="stable")
grp_sorted = grp[order]
offense_result = poss["rate_raw"].to_numpy() / 100
chance = []
for _ in range(B):
    new = np.empty(n)
    new[order] = offense_result[order][np.argsort(grp_sorted + rng.random(n))]
    xs = new * sign
    chance += [z for g0, L in zip(game_start, game_len) for z in scan(xs[g0:g0 + L], live=False)]
chance = np.array(chance) if B else np.array([np.nan])
Z95 = np.quantile(chance, 0.95) if B else np.nan

print(f"{n:,} possessions in {G:,} games.  One push per team per game: {len(pushes):,} pushes.\n")
print("A. THE PUSHES")
print(f"   A typical push: {pushes['pts_for'].median():.0f}-{pushes['pts_against'].median():.0f} over {pushes['n_poss'].median():.0f} possessions "
      f"({pushes['seconds'].median() / 60:.1f} minutes), {pushes['gain'].median():+.1f} points better than expected.")
if B:
    print(f"   Strength of a team's biggest push: real games {pushes['strength'].mean():.2f} on average, shuffled games {chance.mean():.2f} "
          f"({pushes['strength'].mean() / chance.mean() - 1:+.1%}).")
    print(f"   Pushes stronger than 95% of what chance produces (strength above {Z95:.2f}): {(pushes['strength'] > Z95).mean():.1%} of real pushes "
          f"(5.0% would be expected with no momentum at all).")
spread = lambda v: 100 * np.sqrt(max(v, 0.0))                # variance -> points per 100 possessions
weight = lambda v: 0.0 if v <= 0 else np.median(game_len) / (np.median(game_len) + SIGMA ** 2 / v)
print(f"\n   Normal level, or how the game has been going?  First half against second half of the same game (both vs "
      f"expectation): correlation {halves[1].corr(halves[2]):+.3f}.")
print(f"   True night-to-night spread of a team's level: {spread(TONIGHT_VAR):.1f} points per 100 (at most {spread(TONIGHT_VAR_HI):.1f}); "
      f"luck alone moves one game by {100 * SIGMA / np.sqrt(np.median(game_len)):.1f}.")
print(f"   So the game so far deserves a weight of {weight(TONIGHT_VAR):.1%} (at most {weight(TONIGHT_VAR_HI):.1%}) even after a whole game: "
      f"{'pushes are measured against the normal level.' if weight(TONIGHT_VAR_HI) < 0.10 else 'THE GAME SO FAR MATTERS HERE - blend it into the baseline.'}")
big = pushes.sort_values("strength", ascending=False).head(5)
print("\n   The five strongest pushes in the data:")
for _, r in big.iterrows():
    i = int(r["start"])
    who = poss["home"].iloc[i] if r["side"] == 1 else poss["away"].iloc[i]
    other = poss["away"].iloc[i] if r["side"] == 1 else poss["home"].iloc[i]
    print(f"     {poss['gamedate'].iloc[i]}  {who} {r['pts_for']:.0f}-{r['pts_against']:.0f} vs {other}, "
          f"{r['seconds'] / 60:.1f} minutes starting Q{period[i]} {int(clock[i]) // 60}:{int(clock[i]) % 60:02d}  (strength {r['strength']:.1f})")

# ============================================================== B. players
row_of = pd.Series(pos, index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
ev["i"] = row_of.reindex(pd.MultiIndex.from_frame(ev[["game_id", "poss_num"]].astype("int64"))).to_numpy()
ev = ev.dropna(subset=["i"]).astype({"i": int}).reset_index(drop=True)
text = ev["home_txt"].fillna("") + " | " + ev["away_txt"].fillna("")
ei, etype = ev["i"].to_numpy(), ev["etype"].to_numpy()
pid1, pid2, pid3 = (ev[c].fillna(0).to_numpy().astype("int64") for c in ("p1", "p2", "p3"))
p1_side = np.where(ev["t1"].to_numpy() == poss["home"].to_numpy()[ei], 1, -1)
has_team1 = ev["t1"].notna().to_numpy()
# offense: the possession he finished (shot, free throw, turnover) or assisted on
op = pd.concat([pd.DataFrame({"pid": pid1, "i": ei})[np.isin(etype, [1, 2, 3, 5])],
                pd.DataFrame({"pid": pid2, "i": ei})[(etype == 1) & (pid2 > 0)]]).drop_duplicates()
# defense: the possession he ended with a steal, a block or a defensive rebound
dp = pd.concat([pd.DataFrame({"pid": pid2, "i": ei})[(etype == 5) & text.str.contains("STEAL").to_numpy()],
                pd.DataFrame({"pid": pid3, "i": ei})[(etype == 2) & text.str.contains("BLOCK").to_numpy()],
                pd.DataFrame({"pid": pid1, "i": ei})[(etype == 4) & (p1_side == -sign[ei]) & has_team1]]).drop_duplicates()
real_player = lambda d: d[(d["pid"] > 0) & (d["pid"] < 100_000_000)]
op, dp = real_player(op), real_player(dp)
op = op.assign(side=sign[op["i"].to_numpy()], value=(x * sign)[op["i"].to_numpy()])            # his team = the offense
dp = dp.assign(side=-sign[dp["i"].to_numpy()], value=(-x * sign)[dp["i"].to_numpy()])          # his team = the defense
credit = pd.concat([op, dp])                               # every credited play: player, possession, his side, points gained

push_of = {1: np.full(n, -1), -1: np.full(n, -1)}           # which push (if any) each possession belongs to, per side
for pid_, (s, a, b) in enumerate(zip(pushes["side"], st, en)):
    push_of[s][a:b + 1] = pid_
credit["push"] = np.where(credit["side"] == 1, push_of[1][credit["i"].to_numpy()], push_of[-1][credit["i"].to_numpy()])
inside = credit[credit["push"] >= 0]
per_player = inside.groupby(["push", "pid"])["value"].sum().reset_index()
leaders = per_player.loc[per_player.groupby("push")["value"].idxmax()]
leaders = leaders[leaders["value"] > 0][["push", "pid"]]
starters = inside[inside["i"].to_numpy() == st[inside["push"].to_numpy()]][["push", "pid"]].drop_duplicates()
# the play that ended it: the first possession after the push, credited to the OTHER team's players
after = en + 1
has_after = (after < n) & (gidx[np.minimum(after, n - 1)] == gidx[en])
end_row = pd.Series(np.where(has_after, after, -1))
enders = credit.merge(pd.DataFrame({"push": np.arange(len(pushes)), "i": end_row, "side": -pushes["side"].to_numpy()}),
                      on=["i", "side"], suffixes=("_own", ""))[["push", "pid"]].drop_duplicates()

off_five = poss[[f"o{k}" for k in range(1, 6)]].to_numpy()
def_five = poss[[f"d{k}" for k in range(1, 6)]].to_numpy()
complete = (poss["complete"] == 1).to_numpy()


def five(rows_, side):
    """The five players of `side` on the floor at each of the given possessions."""
    return np.where((sign[rows_] == side)[:, None], off_five[rows_], def_five[rows_])


# Starters = the five on the floor for each team at the game's opening possession. Everyone else who
# plays in that game is a BENCH player for that game. (Decided game by game, so no cut-off such as
# "started fewer than half his games" is needed.)
opening_known = complete[game_start]
starter_keys = np.unique(np.r_[(np.arange(G)[:, None] * 100_000_000 + five(game_start, np.ones(G, int))).ravel(),
                               (np.arange(G)[:, None] * 100_000_000 + five(game_start, -np.ones(G, int))).ravel()])


def player_table(events, floor_rows, floor_side, title, what, bench=False):
    """Rate = pushes in which he did it / pushes while he was on the floor, bigger pushes counting more.
    bench=True keeps only games in which the player did not start."""
    use = (floor_rows >= 0) & complete[np.maximum(floor_rows, 0)]
    w = pushes["gain"].to_numpy()
    on = pd.DataFrame({"push": np.repeat(np.flatnonzero(use), 5), "pid": five(floor_rows[use], floor_side[use]).ravel()})
    on["w"], on["season"] = w[on["push"].to_numpy()], pushes["season"].to_numpy()[on["push"].to_numpy()]
    on = on[playoffs[st[on["push"].to_numpy()]] == SCOPE]
    if bench:
        g_ = pushes["g"].to_numpy()[on["push"].to_numpy()]
        on = on[opening_known[g_] & ~np.isin(g_ * 100_000_000 + on["pid"].to_numpy(), starter_keys)]
    did = events.assign(did=1.0)
    on = on.merge(did, on=["push", "pid"], how="left").fillna({"did": 0.0})
    on = on.assign(dw=on["did"] * on["w"], w2=on["w"] ** 2)
    g = on.groupby(["pid", "season"])
    ps = pd.DataFrame({"pushes": g.size(), "rate": g["dw"].sum() / g["w"].sum(), "n_eff": g["w"].sum() ** 2 / g["w2"].sum()}).reset_index()
    league = on["dw"].sum() / on["w"].sum()
    ps["se"] = np.sqrt(league * (1 - league) / ps["n_eff"])                     # pooled, so few pushes never look precise
    # one season vs the next
    pairs = ps.merge(ps.assign(season=ps["season"] - 1), on=["pid", "season"], suffixes=("", "_next"))
    if len(pairs) >= 30:
        wts = 1 / (pairs["se"] ** 2 + pairs["se_next"] ** 2)
        a, b = pairs["rate"].to_numpy(), pairs["rate_next"].to_numpy()
        corr = lambda u, v: np.cov(u, v, aweights=wts)[0, 1] / np.sqrt(np.cov(u, u, aweights=wts)[0, 1] * np.cov(v, v, aweights=wts)[0, 1])
        r = corr(a, b)
        luck = np.array([corr(a, rng.permutation(b)) for _ in range(1000)])
        verdict = "a lasting trait" if r > np.quantile(luck, 0.99) else "weak hint" if r > np.quantile(luck, 0.95) else "no - looks like luck"
        rep = f"{r:+.2f} over {len(pairs):,} pairs of seasons ({verdict})"
    else:
        rep = "not enough seasons to say"
    # average of his seasons (by precision), pulled toward the league average by how much is luck
    wt = 1 / ps["se"] ** 2
    g = ps.assign(wt=wt, vw=ps["rate"] * wt).groupby("pid")
    avg = pd.DataFrame({"seasons": g.size(), "pushes a season": g["pushes"].mean(), "rate": g["vw"].sum() / g["wt"].sum(),
                        "se": 1 / np.sqrt(g["wt"].sum())})
    real_var = max(np.average((avg["rate"] - league) ** 2, weights=1 / avg["se"] ** 2) - np.average(avg["se"] ** 2, weights=1 / avg["se"] ** 2), 0.0)
    avg["fair estimate"] = league + real_var / (real_var + avg["se"] ** 2) * (avg["rate"] - league)
    # For the verdict, compare on the arcsine-square-root scale: with few pushes a share is lopsided
    # (it cannot go below zero), and on this scale luck behaves evenly in both directions.
    n_total = league * (1 - league) / avg["se"] ** 2
    z = 2 * (np.arcsin(np.sqrt(avg["rate"].clip(0, 1))) - np.arcsin(np.sqrt(league))) * np.sqrt(n_total)
    avg["verdict"] = np.select([z > 1.96, z < -1.96], ["clearly above average", "clearly below average"], "can't tell from average")
    avg["player"], avg["+/-"], avg["evidence"] = names.reindex(avg.index).to_numpy(), 1.96 * avg["se"], z
    avg = avg.sort_values(["fair estimate", "evidence"], ascending=False)
    print(f"\n{title}")
    print(f"  {what}")
    print(f"  Repeats?  {rep}")
    print(f"  {len(avg):,} players; league average {league:.1%}; real spread between players {np.sqrt(real_var):.1%}; clearly above average: "
          f"{(avg['verdict'] == 'clearly above average').sum()}, clearly below: {(avg['verdict'] == 'clearly below average').sum()} "
          f"(luck alone would flag about {0.025 * len(avg):.0f} each way)")
    cols = ["player", "seasons", "pushes a season", "rate", "+/-", "fair estimate", "verdict"]
    fmt = {"pushes a season": "{:.0f}".format, "rate": "{:.1%}".format, "+/-": "{:.1%}".format, "fair estimate": "{:.1%}".format}
    print(avg[cols].head(TOP).to_string(index=False, formatters=fmt))
    return avg.reset_index().assign(measure=title.split(" - ")[0], games="playoffs" if SCOPE else "regular season")[["games", "measure", "pid", *cols]]


# ---------------------------------------------------------------- his usual share of the plays
# For every possession and each of the five players of a team on the floor: was he credited, and what
# was the chance he would be, going by his usual? "Usual" = how often he is credited on that KIND of
# possession over the whole season (regular season and playoffs kept apart). The kind matters because
# a push is full of made baskets: it has to be compared with his usual share of made baskets.
off_points = poss["points"].to_numpy()
all_pids = np.unique(np.r_[off_five[complete].ravel(), def_five[complete].ravel()])
scope_no = pd.factorize(season * 2 + playoffs)[0]            # a season's regular season and its playoffs are separate
N_SCOPE, N_KIND = scope_no.max() + 1, 6


def usual(fives, credits, kind):
    """fives: the five players at every possession. credits: (player, possession) pairs credited.
    kind: what kind of possession each one is. Returns who was credited and each player's usual chance."""
    did = np.zeros(fives.shape, bool)
    ci, cp = credits["i"].to_numpy(), credits["pid"].to_numpy()
    r, c = np.nonzero(fives[ci] == cp[:, None])               # the credited player's seat among the five
    did[ci[r], c] = True
    did &= complete[:, None]
    key = (np.searchsorted(all_pids, fives).clip(0, len(all_pids) - 1) * N_SCOPE + scope_no[:, None]) * N_KIND + kind[:, None]
    size = len(all_pids) * N_SCOPE * N_KIND
    k_ok = key[complete].ravel()
    credited = np.bincount(k_ok, weights=did[complete].ravel(), minlength=size)
    rate = (credited / np.maximum(np.bincount(k_ok, minlength=size), 1))[key]       # his usual rate on this kind of possession
    n_credits = did.sum(axis=1)[:, None]                                            # how many credits were handed out here
    share = np.where(rate.sum(axis=1)[:, None] > 0, rate / np.maximum(rate.sum(axis=1)[:, None], 1e-12), 0.2)
    chance = n_credits * share                                                       # shared out among these five
    # scale each player's chances so that over his season they add up to what he actually did (per kind)
    expected = np.bincount(k_ok, weights=chance[complete].ravel(), minlength=size)
    scale = np.where(expected > 0, credited / np.maximum(expected, 1e-12), 0.0)
    return did, np.minimum(chance * scale[key], 1.0)


did_off, usual_off = usual(off_five, op, np.minimum(off_points, 3).astype(int))     # scored 0, 1, 2 or 3+
did_def, usual_def = usual(def_five, dp, np.where(off_points == 0, 4, 5))            # a stop, or not


def on_side(rows_, side):
    """For the five players of `side` at the given possessions: who they are, who was credited, each one's usual chance."""
    off = (sign[rows_] == side)[:, None]
    return (np.where(off, off_five[rows_], def_five[rows_]), np.where(off, did_off[rows_], did_def[rows_]),
            np.where(off, usual_off[rows_], usual_def[rows_]))


def usual_table(ps, title, what, phi):
    """ps: one row per player-season with actual, usual, the standard error of the difference and n = how much
    there was to go on. Seasons and players are weighted by n (how much he played), not by 1 / se^2: a
    player with two plays and a tiny usual chance has a tiny se and would otherwise outweigh everybody."""
    ps = ps[np.isfinite(ps["se"]) & (ps["se"] > 0) & (ps["n"] > 0)].assign(diff=lambda d: d["actual"] - d["usual"])
    pairs = ps.merge(ps.assign(season=ps["season"] - 1), on=["pid", "season"], suffixes=("", "_next"))
    if len(pairs) >= 30:
        wts = 1 / (1 / pairs["n"] + 1 / pairs["n_next"])
        a, b = pairs["diff"].to_numpy(), pairs["diff_next"].to_numpy()
        corr = lambda u, v: np.cov(u, v, aweights=wts)[0, 1] / np.sqrt(np.cov(u, u, aweights=wts)[0, 1] * np.cov(v, v, aweights=wts)[0, 1])
        r = corr(a, b)
        luck = np.array([corr(a, rng.permutation(b)) for _ in range(1000)])
        verdict = "a lasting trait" if r > np.quantile(luck, 0.99) else "weak hint" if r > np.quantile(luck, 0.95) else "no - looks like luck"
        rep = f"{r:+.2f} over {len(pairs):,} pairs of seasons ({verdict})"
    else:
        rep = "not enough seasons to say"
    wt = ps["n"]
    g = ps.assign(wt=wt, aw=ps["actual"] * wt, uw=ps["usual"] * wt, dw=ps["diff"] * wt, vw=(ps["se"] * wt) ** 2).groupby("pid")
    avg = pd.DataFrame({"seasons": g.size(), "pushes a season": g["pushes"].mean(), "in pushes": g["aw"].sum() / g["wt"].sum(),
                        "his usual": g["uw"].sum() / g["wt"].sum(), "difference": g["dw"].sum() / g["wt"].sum(),
                        "se": np.sqrt(g["vw"].sum()) / g["wt"].sum(), "n": g["wt"].sum()})
    # real spread between players = what is left of the spread of the differences once luck (se^2) is taken out
    real_var = max(np.average(avg["difference"] ** 2 - avg["se"] ** 2, weights=avg["n"]), 0.0)
    avg["fair estimate"] = real_var / (real_var + avg["se"] ** 2) * avg["difference"]
    # A count moves in whole plays, so half a play is taken off before judging (the "continuity correction").
    # With many plays this changes nothing; with two or three it stops one lucky play from looking like proof.
    typical = np.average(avg["his usual"], weights=avg["n"])
    plays = typical * (1 - typical) / avg["se"] ** 2                              # how many plays the estimate rests on
    z = np.sign(avg["difference"]) * np.maximum(avg["difference"].abs() - 0.5 / plays, 0) / avg["se"]
    avg["verdict"] = np.select([z > 1.96, z < -1.96], ["clearly more than his usual", "clearly less than his usual"], "can't tell from his usual")
    avg["player"], avg["+/-"], avg["evidence"] = names.reindex(avg.index).to_numpy(), 1.96 * avg["se"], z
    avg = avg.sort_values(["fair estimate", "evidence"], ascending=False)
    print(f"\n{title}")
    print(f"  {what}")
    print(f"  Repeats?  {rep}")
    print(f"  {len(avg):,} players; real spread between players {np.sqrt(real_var):.1%}; clearly more than usual: "
          f"{(avg['verdict'] == 'clearly more than his usual').sum()}, clearly less: {(avg['verdict'] == 'clearly less than his usual').sum()} "
          f"(luck alone would flag about {0.025 * len(avg):.0f} each way); plays in the same push move together x{phi:.2f}")
    cols = ["player", "seasons", "pushes a season", "in pushes", "his usual", "difference", "+/-", "fair estimate", "verdict"]
    pct = "{:.1%}".format
    signed = "{:+.1%}".format
    fmt = {"pushes a season": "{:.0f}".format, "in pushes": pct, "his usual": pct, "difference": signed, "+/-": pct, "fair estimate": signed}
    print(avg[cols].head(TOP).to_string(index=False, formatters=fmt))
    return avg.reset_index().assign(measure=title.split(" - ")[0], games="playoffs" if SCOPE else "regular season")[["games", "measure", "pid", *cols]]


def usual_rate(rows_, side_, title, what):
    """One play per push (the one that started it, or the one that ended it): how often it was his,
    against how often his usual share of such plays says it would be. Bigger pushes count more."""
    use = (rows_ >= 0) & complete[np.maximum(rows_, 0)] & (playoffs[st] == SCOPE)
    who, did, chance = on_side(rows_[use], side_[use])
    w = np.repeat(pushes["gain"].to_numpy()[use], 5)
    d = pd.DataFrame({"pid": who.ravel(), "season": np.repeat(pushes["season"].to_numpy()[use], 5), "w": w, "a": w * did.ravel(),
                      "e": w * chance.ravel(), "v": w ** 2 * (chance * (1 - chance)).ravel(), "r2": w ** 2 * ((did - chance) ** 2).ravel(),
                      "w2": w ** 2})
    phi = d["r2"].sum() / d["v"].sum()                          # 1 = exactly coin-flip luck
    g = d.groupby(["pid", "season"])
    ps = pd.DataFrame({"pushes": g.size(), "actual": g["a"].sum() / g["w"].sum(), "usual": g["e"].sum() / g["w"].sum(),
                       "se": np.sqrt(phi * g["v"].sum()) / g["w"].sum(), "n": g["w"].sum() ** 2 / g["w2"].sum()}).reset_index()
    return usual_table(ps, title, what, phi)


def usual_share(title, what):
    """All the plays of his team's pushes while he was on the floor: his share of the credit, against his usual share."""
    lens = en - st + 1
    rows_ = np.repeat(st, lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)      # every possession of every push
    push_, side_ = np.repeat(np.arange(len(pushes)), lens), np.repeat(pushes["side"].to_numpy(), lens)
    use = complete[rows_] & (playoffs[st][push_] == SCOPE)
    rows_, push_, side_ = rows_[use], push_[use], side_[use]
    who, did, chance = on_side(rows_, side_)
    v = (x[rows_] * side_)[:, None]                             # what the possession was worth to the pushing team
    d = pd.DataFrame({"pid": who.ravel(), "push": np.repeat(push_, 5), "a": (v * did).ravel(), "e": (v * chance).ravel(),
                      "d": np.repeat(v[:, 0] * did.sum(axis=1), 5), "v": (v ** 2 * chance * (1 - chance)).ravel()})
    per_push = d.groupby(["pid", "push"], sort=False)[["a", "e", "d", "v"]].sum().reset_index()
    phi = ((per_push["a"] - per_push["e"]) ** 2).sum() / per_push["v"].sum()
    per_push["season"] = pushes["season"].to_numpy()[per_push["push"].to_numpy()]
    g = per_push.groupby(["pid", "season"])
    tot = g[["a", "e", "d", "v"]].sum().assign(pushes=g.size()).reset_index()
    tot = tot[tot["d"] > 0]                                     # a share needs some credit to be a share of
    ps = pd.DataFrame({"pid": tot["pid"], "season": tot["season"], "pushes": tot["pushes"], "actual": tot["a"] / tot["d"],
                       "usual": tot["e"] / tot["d"], "se": np.sqrt(phi * tot["v"]) / tot["d"], "n": tot["d"]})
    return usual_table(ps, title, what, phi)


saved, saved_usual = [], []
for SCOPE, label in ((False, "REGULAR SEASON"), (True, "PLAYOFFS")):
    if not (playoffs == SCOPE).any():
        continue
    print(f"\n\nB. PLAYERS - {label}  (rate = share of the pushes that happened while he was on the floor; bigger pushes count more)")
    print("   These lists measure involvement: whoever is involved in most plays is involved in most pushes, and their")
    print("   season-to-season repeatability is the repeatability of involvement. B3 asks whether anyone does more than his usual.")
    side_ = pushes["side"].to_numpy()
    saved.append(player_table(starters, st, side_, "STARTS - credited on the first possession of his team's push (involvement)",
                              "any play credited to him on that possession: a shot (made or missed), free throws, a turnover, an assist, a steal, a block or a defensive rebound"))
    saved.append(player_table(leaders, st, side_, "LEADS - most credit within his team's push (involvement)",
                              "points better than expected on plays he finished or assisted, plus points saved on defense; "
                              "if all five shared equally this would be 20%"))
    saved.append(player_table(enders, end_row.to_numpy(), -side_, "STOPS - credited on the possession right after the other team's push (involvement)",
                              "any play credited to him on that possession: a shot (made or missed), free throws, a turnover, an assist, a steal, a block or a defensive rebound"))
    print(f"\n\nB2. BENCH PLAYERS - {label}  (only the games in which he did NOT start; compared with other bench players)")
    saved.append(player_table(starters, st, side_, "BENCH STARTS - off the bench, credited on the first possession of his team's push (involvement)",
                              "any play credited to him on that possession: a shot (made or missed), free throws, a turnover, an assist, a steal, a block or a defensive rebound", bench=True))
    saved.append(player_table(leaders, st, side_, "BENCH LEADS - off the bench, most credit within his team's push (involvement)",
                              "points better than expected on plays he finished or assisted, plus points saved on defense", bench=True))
    saved.append(player_table(enders, end_row.to_numpy(), -side_, "BENCH STOPS - off the bench, credited right after the other team's push (involvement)",
                              "any play credited to him on that possession: a shot (made or missed), free throws, a turnover, an assist, a steal, a block or a defensive rebound", bench=True))
    print(f"\n\nB3. MORE THAN HIS USUAL - {label}  (the same plays, compared with the player's own usual share of such plays; "
          f"positive = he does more during pushes than he normally does)")
    saved_usual.append(usual_rate(st, side_, "STARTS MORE THAN HIS USUAL - made the first play of his team's push more often than his usual share says",
                                  "in pushes = how often the first play was his; his usual = how often it would be, going by his usual share of such plays"))
    saved_usual.append(usual_share("CARRIES MORE THAN HIS USUAL - takes a bigger share of his team's push than his usual share",
                                   "in pushes = his share of all the credit in his team's pushes; if the five shared equally this would be 20%"))
    saved_usual.append(usual_rate(end_row.to_numpy(), -side_, "STOPS MORE THAN HIS USUAL - made the play that ended the other team's push more often than his usual share says",
                                  "in pushes = how often the ending play was his; his usual = how often it would be, going by his usual share of such plays"))

def save_players():
    if DB == "data/nba.db":
        Path("results").mkdir(exist_ok=True)
        pd.concat(saved).to_csv("results/push_players.csv", index=False, float_format="%.4f")
        pd.concat(saved_usual).to_csv("results/push_players_usual.csv", index=False, float_format="%.4f")
        print("\nSaved results/push_players.csv and results/push_players_usual.csv")


if SKIP_TIMEOUTS:
    save_players()
    raise SystemExit

# ============================================================== C. timeouts and the push in progress
print("\n\nC. TIMEOUTS WHILE A PUSH IS UNDER WAY  (coach-chosen timeouts only; forced TV slots left out)")
timeouts = timeouts[timeouts["game_id"].isin(np.unique(game))].copy()
timeouts["season"] = timeouts["game_id"].map(pd.Series(season, index=game).groupby(level=0).first())
timeouts = timeouts.sort_values(["game_id", "period", "seconds_left"], ascending=[True, True, False])
taken = timeouts.groupby(["game_id", "period"]).cumcount().to_numpy()
t_clock, t_period = timeouts["seconds_left"].to_numpy(), timeouts["period"].to_numpy()
required = np.where(timeouts["season"].to_numpy() >= 2017, (t_clock <= 419) * 1 + (t_clock <= 179),
                    (t_clock <= 359) * 1 + (t_clock <= 179) + ((t_clock <= 539) & np.isin(t_period, [2, 4])))
forced = (taken < required) & (t_period <= 4)
game_no = pd.Series(np.arange(G), index=pd.unique(game))
key = gidx * 10_000 + period * 1_000 + (720 - clock)
t_key = game_no.reindex(timeouts["game_id"]).to_numpy() * 10_000 + t_period * 1_000 + (720 - t_clock)
tj = np.minimum(np.searchsorted(key, t_key, side="left"), n - 1)
good = (game[tj] == timeouts["game_id"].to_numpy()) & (period[tj] == t_period)
after_timeout = poss["starttype"].fillna("").to_numpy() == "Off Timeout"
# A timeout whose clock falls inside the window of an "Off Timeout" possession belongs to that possession (the
# possession is dated from the stoppage before it). Skipping these used to drop about half of the timeouts.
j0 = np.maximum(tj - 1, 0)
inside = ((game[j0] == timeouts["game_id"].to_numpy()) & (period[j0] == t_period) & (tj > 0) & (clock[j0] > t_clock)
          & (t_clock >= poss["end_seconds_left"].to_numpy()[j0]))
use_prev = after_timeout[j0] & inside & ~(good & (clock[tj] == t_clock) & after_timeout[tj])
tj, good = np.where(use_prev, j0, tj), good | use_prev
called_by, is_forced = np.zeros(n), np.zeros(n, bool)
# who called it: the team named in its own description column. "Timeout: Official" (2015-17) and a few "Timeout:
# Regular" lines name no team - an official's stoppage, nobody's choice - so they are left out like the TV slots.
team_named = (timeouts["home_txt"].notna() | timeouts["away_txt"].notna()).to_numpy()
called_by[tj[good]] = np.where(timeouts["home_txt"].notna(), 1, np.where(timeouts["away_txt"].notna(), -1, 0))[good]
np.logical_or.at(is_forced, tj[good], (forced | ~team_named)[good])

# how much FURTHER each side's push goes from every moment (points vs expectation still to be gained before its peak)
c_before = cx[:-1]                                                       # home's running total before each possession
further = {1: np.zeros(n), -1: np.zeros(n)}
for g0, L in zip(game_start, game_len):
    path = np.r_[c_before[g0:g0 + L], cx[g0 + L]]                         # home's running total at every point of the game
    further[1][g0:g0 + L] = np.maximum.accumulate(path[::-1])[::-1][:L] - path[:L]
    further[-1][g0:g0 + L] = path[:L] - np.minimum.accumulate(path[::-1])[::-1][:L]

# one row per (stoppage, team): the team's view of the OPPONENT's push in progress
usable = ~first_of_period & ~(after_timeout & (called_by == 0)) & ~is_forced
rows_ = np.flatnonzero(usable)
R = len(rows_)
idx = np.r_[rows_, rows_]
has_ball = np.r_[np.ones(R, bool), np.zeros(R, bool)]
side = np.where(has_ball, sign[idx], -sign[idx])                          # this team: +1 home / -1 away
opp_strength = np.where(side == 1, live_strength[-1][idx], live_strength[1][idx])        # opponent's push in progress
opp_started = np.where(side == 1, live_start[-1][idx], live_start[1][idx])
opp_further = np.where(side == 1, further[-1][idx], further[1][idx])                     # how much further it goes
so_far_opp = np.where(side == 1, cap[idx] - cap[opp_started], chp[idx] - chp[opp_started])   # the push so far, in real points
so_far_own = np.where(side == 1, chp[idx] - chp[opp_started], cap[idx] - cap[opp_started])
flip = np.where(has_ball, 1.0, -1.0)
net_ahead = (poss["ahead_own"] - poss["ahead_opp"]).to_numpy()[idx] * flip               # this team's momentum ahead (per 100)
mom = (poss["before_off"] - poss["before_def"]).to_numpy()[idx] * flip
treated = after_timeout[idx] & (called_by[idx] == side)
keep = ~(after_timeout[idx] & (called_by[idx] == -side))
g_row = gidx[idx]
OUT = {"how much further the opponent's push goes (points)": opp_further, "his team's momentum afterwards (per 100)": net_ahead}


def spline(v):
    k = np.unique(np.percentile(v, KNOT_PCT))
    if len(k) < 3:
        return v[:, None]
    cube = lambda u: np.maximum(u, 0) ** 3
    return np.column_stack([v] + [(cube(v - k[j]) - cube(v - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                                   + cube(v - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2 for j in range(len(k) - 2)])


prev_end = start_class[idx]                              # how the possession would have started (see would_be_start)
margin = poss["start_margin"].to_numpy(float)[idx] * flip
late = (period[idx] >= 4).astype(float)
X = np.column_stack([np.ones(2 * R), has_ball, side == 1, spline(opp_strength), opp_strength * has_ball, mom / mom.std(),
                     spline(margin), spline(margin) * late[:, None], late, spline(np.log1p(clock[idx])),
                     pd.get_dummies(np.minimum(period[idx], 5)).to_numpy(float)[:, 1:],
                     pd.get_dummies(prev_end).to_numpy(float)[:, 1:], pd.get_dummies(prev_end).to_numpy(float)[:, 1:] * has_ball[:, None],
                     playoffs[idx]]).astype(float)
X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / np.where(X[:, 1:].std(0) > 0, X[:, 1:].std(0), 1)
fold = g_row % FOLDS
propensity = np.zeros(2 * R)
expected0 = {name: np.zeros(2 * R) for name in OUT}
for f in range(FOLDS):                      # both models are fitted without the moment's own games
    fit, out_ = keep & (fold != f), fold == f
    beta = np.zeros(X.shape[1])
    Xf, tf = X[fit], treated[fit].astype(float)
    for _ in range(25):                      # chance of a timeout in this situation (logistic regression)
        p_ = 1 / (1 + np.exp(-np.clip(Xf @ beta, -30, 30)))
        step = np.linalg.solve((Xf * (p_ * (1 - p_))[:, None]).T @ Xf + 1e-6 * np.eye(X.shape[1]), Xf.T @ (tf - p_))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    propensity[out_] = 1 / (1 + np.exp(-np.clip(X[out_] @ beta, -30, 30)))
    c_ = fit & ~treated                      # what happens next without a timeout (linear regression)
    inv = np.linalg.pinv(X[c_].T @ X[c_])
    for name, y in OUT.items():
        expected0[name][out_] = X[out_] @ (inv @ (X[c_].T @ y[c_]))
ipw = np.where(treated, 0.0, np.minimum(propensity / (1 - propensity), 50.0)) * keep
resid = {name: OUT[name] - expected0[name] for name in OUT}
control = keep & ~treated


def components(r, m, w):
    """Pooled noise sizes: a game's average = the level + a game-wide part + moment-by-moment noise."""
    d = pd.DataFrame({"g": g_row[m], "xw": r[m] * w, "w": w, "w2": w * w}).groupby("g").sum()
    d = d[d["w"] > 0]
    dev2 = (d["xw"] / d["w"] - d["xw"].sum() / d["w"].sum()) ** 2
    return np.maximum(np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["w2"] / d["w"] ** 2]), dev2, rcond=None)[0], 0)


def effect(mt, mc, name, extra=None, noise=None):
    """Doubly robust effect of the timeouts in mt, against the comparison moments in mc. `extra` = an extra
    weight per moment (e.g. the strength of the push), applied to both groups."""
    r = resid[name]
    e_ = np.ones(2 * R) if extra is None else extra
    wt, wc = e_[mt], ipw[mc] * e_[mc]
    parts = []
    for m_, w_, k_n in ((mt, wt, 0), (mc, wc, 1)):
        if w_.sum() == 0:
            return np.nan, np.nan
        d = pd.DataFrame({"g": g_row[m_], "x": r[m_] * w_, "w": w_, "w2": w_ * w_}).groupby("g").sum()
        game_part, play_part = noise[k_n] if noise else components(r, m_, w_)
        parts.append((d["x"].sum() / d["w"].sum(), (game_part * (d["w"] ** 2).sum() + play_part * d["w2"].sum()) / d["w"].sum() ** 2))
    return parts[0][0] - parts[1][0], np.sqrt(parts[0][1] + parts[1][1])


for SCOPE, label in ((False, "REGULAR SEASON"), (True, "PLAYOFFS")):
    scope = playoffs[idx] == SCOPE
    t_all, c_all = treated & scope, control & scope
    if t_all.sum() < 200:
        continue
    print(f"\n{label}: {t_all.sum():,} coach-chosen timeouts")
    edges = np.quantile(opp_strength[t_all], [0.5, 0.8, 0.95])
    band = np.digitize(opp_strength, edges)
    table = []
    for b_, lab in ((None, "all timeouts"), (0, "weakest half of pushes"), (1, "next 30%"), (2, "next 15%"), (3, "strongest 5% of pushes")):
        m_ = np.ones(2 * R, bool) if b_ is None else band == b_
        mt, mc = t_all & m_, c_all & m_
        row = {"opponent's push when the timeout is called": lab, "timeouts": int(mt.sum()),
               "push so far": f"{so_far_opp[mt].mean():.1f}-{so_far_own[mt].mean():.1f}", "strength": opp_strength[mt].mean()}
        for name, short in zip(OUT, ("change in how much further it goes (points)", "change in his team's momentum (per 100)")):
            est, se = effect(mt, mc, name)
            row[short] = f"{est:+.2f} (+/-{1.96 * se:.2f})"
        row["further without a timeout"] = opp_further[mt].mean() - effect(mt, mc, list(OUT)[0])[0]
        table.append(row)
    print(pd.DataFrame(table).to_string(index=False, formatters={"timeouts": "{:,}".format, "strength": "{:.2f}".format,
                                                                 "further without a timeout": "{:.2f}".format}))
    print("  'change in how much further it goes': NEGATIVE = GOOD (the push is cut short). 'change in momentum': POSITIVE = GOOD.")

    if coach_of is None:
        continue
    team_name = np.where(side == 1, poss["home"].to_numpy()[idx], poss["away"].to_numpy()[idx])
    coach = coach_of.reindex(pd.MultiIndex.from_arrays([game[idx], team_name])).to_numpy()
    cs_code, cs_names = pd.factorize(pd.Series(coach, dtype="object").fillna("?").to_numpy() + "|" + season[idx].astype(str))
    MEASURES = (("cut", list(OUT)[0], -1), ("mom", list(OUT)[1], 1))       # (short name, outcome, sign that is GOOD)
    noise = {short: (components(resid[name], t_all, opp_strength[t_all]), components(resid[name], c_all, (ipw * opp_strength)[c_all]))
             for short, name, _ in MEASURES}
    recs = []
    for code in np.unique(cs_code[t_all & pd.notna(coach)]):
        mk = cs_code == code
        mt, mc = t_all & mk, c_all & mk
        if mt.sum() < 2 or mc.sum() < 2 or opp_strength[mt].sum() == 0:
            continue
        k = cs_names[code]
        rec = {"coach": k.split("|")[0], "season": int(k.split("|")[1]), "timeouts": int(mt.sum()), "strength": opp_strength[mt].mean()}
        for short, name, _ in MEASURES:
            rec[short], rec[short + "_se"] = effect(mt, mc, name, extra=opp_strength, noise=noise[short])
        recs.append(rec)
    cs = pd.DataFrame(recs).dropna()
    pairs = cs.merge(cs.assign(season=cs["season"] - 1), on=["coach", "season"], suffixes=("", "_next"))
    print("\n  COACHES - what his timeouts do to the opponent's push. Timeouts called during stronger pushes count more.")
    print("  Each coach is compared with his own team's no-timeout moments.")
    print("    rest of push   = how much further the opponent's push goes (points).   NEGATIVE = GOOD (cut short)")
    print("    momentum after = his team's momentum over the next possessions (per 100).   POSITIVE = GOOD")
    if len(pairs) >= 20:
        for col, lab in (("strength", "how strong the push is when he calls one"), ("cut", "effect on the rest of the push"),
                         ("mom", "effect on his team's momentum after")):
            a, b = pairs[col].to_numpy(), pairs[f"{col}_next"].to_numpy()
            r = np.corrcoef(a, b)[0, 1]
            luck = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(5000)])
            verdict = "a lasting trait" if r > np.quantile(luck, 0.99) else "weak hint" if r > np.quantile(luck, 0.95) else "no - looks like luck"
            print(f"  Repeats?  {lab:42s} {r:+.2f} over {len(pairs)} pairs of seasons ({verdict})")
    g = cs.groupby("coach")
    board = pd.DataFrame({"seasons": g.size(), "timeouts": g["timeouts"].sum(), "push strength when he calls it": g["strength"].mean()})
    spread, centres = {}, {}
    for short, _, good_sign in MEASURES:
        wt = 1 / cs[short + "_se"] ** 2
        gw = cs.assign(wt=wt, vw=cs[short] * wt).groupby("coach")
        board[short], board[short + "_se"] = gw["vw"].sum() / gw["wt"].sum(), 1 / np.sqrt(gw["wt"].sum())
        w_ = 1 / board[short + "_se"] ** 2
        centre = np.average(board[short], weights=w_)
        real_var = max(np.average((board[short] - centre) ** 2, weights=w_) - np.average(board[short + "_se"] ** 2, weights=w_), 0.0)
        spread[short], centres[short] = np.sqrt(real_var), centre
        board[short + "_fair"] = centre + real_var / (real_var + board[short + "_se"] ** 2) * (board[short] - centre)
        # each coach against the AVERAGE coach: the league-wide effect is not zero, so testing against zero would flag
        # coaches for the league-wide effect rather than for anything of their own
        z = good_sign * (board[short] - centre) / board[short + "_se"]
        board[short + "_verdict"] = np.select([z > 1.96, z < -1.96], ["clearly better than average", "clearly worse than average"],
                                              "can't tell")
        board[short + "_pm"] = 1.96 * board[short + "_se"]
    board["evidence"] = (board["mom"] - centres["mom"]) / board["mom_se"]
    board = board.sort_values(["mom_fair", "evidence"], ascending=False)
    flagged = {short: (board[short + "_verdict"] != "can't tell").sum() for short, _, _ in MEASURES}
    print(f"  {len(board)} coaches, ranked by the fair estimate of 'momentum after' (the more sensitive of the two measures).")
    print(f"  Real spread between coaches: rest of push {spread['cut']:.2f} points, momentum after {spread['mom']:.2f}. "
          f"Average coach: rest of push {centres['cut']:+.2f}, momentum after {centres['mom']:+.1f}.")
    print(f"  Clearly better or worse than the average coach: {flagged['cut']} on rest of push, {flagged['mom']} on momentum after "
          f"(luck alone would flag about {0.05 * len(board):.0f} on each).")
    show = board.reset_index()[["coach", "seasons", "timeouts", "push strength when he calls it", "cut", "cut_pm", "cut_verdict",
                                "mom", "mom_pm", "mom_fair", "mom_verdict"]]
    show.columns = ["coach", "seasons", "timeouts", "push strength when he calls it", "rest of push", "+/-", "verdict",
                    "momentum after", "+/- ", "fair estimate", "verdict "]
    print(show.to_string(index=False, formatters={"timeouts": "{:,}".format, "push strength when he calls it": "{:.2f}".format,
                                                  "rest of push": "{:+.2f}".format, "+/-": "{:.2f}".format,
                                                  "momentum after": "{:+.1f}".format, "+/- ": "{:.1f}".format,
                                                  "fair estimate": "{:+.1f}".format}))
    if DB == "data/nba.db":
        Path("results").mkdir(exist_ok=True)
        show.to_csv(f"results/push_coaches_{'playoffs' if SCOPE else 'regular'}.csv", index=False, float_format="%.3f")

save_players()
if DB == "data/nba.db":
    out_p = pushes.assign(team=np.where(pushes["side"] == 1, poss["home"].to_numpy()[st], poss["away"].to_numpy()[st]),
                          game_id=game[st], date=poss["gamedate"].to_numpy()[st])
    out_p[["game_id", "date", "team", "pts_for", "pts_against", "n_poss", "seconds", "gain", "strength"]].to_csv(
        "results/pushes.csv", index=False, float_format="%.3f")
    print("Saved results/pushes.csv, push_coaches_*.csv")
