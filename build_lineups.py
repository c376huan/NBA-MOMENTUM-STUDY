"""
NBA Momentum Study - Step 3b: who is on the floor for every possession.

The play-by-play lists substitutions ("SUB: A FOR B") but never the five
players who START each period. We work them out:

  1. Walk through a period's plays in order. A player is a starter if he
     does something (shoots, rebounds, fouls, is fouled, gets subbed OUT...)
     before he was ever subbed IN during that period.
  2. A starter who plays the whole period without a single recorded play and
     without being substituted leaves no trace. If a team is short of five,
     we fill from the players who finished the previous period, when exactly
     the right number of them are unaccounted for. Otherwise the lineup is
     marked incomplete and those possessions are not used by the models that
     need lineups.
  3. Replay the period applying each substitution, and give every possession
     the ten players on the floor when it starts.

Technical fouls are ignored as evidence (they can be called on the bench).

Output: table `lineups` - game_id, poss_num, o1..o5 (offense player ids),
d1..d5 (defense), complete (1/0) - and table `players` (id, name).

Checks printed at the end: share of possessions with a complete lineup, and
how often the player who scored is among the five we think are on offense.

Usage:  python build_lineups.py [--db data/fake_nba.db]
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "data/nba.db"
con = sqlite3.connect(DB)

ev = pd.read_sql_query(
    """SELECT rowid AS rid, game_id, period, seconds_left, eventmsgtype AS t, eventmsgactiontype AS a,
              COALESCE(homedescription, '') || ' ' || COALESCE(visitordescription, '') AS txt,
              player1_id AS p1, player1_name AS n1, player1_team_abbreviation AS t1,
              player2_id AS p2, player2_name AS n2, player2_team_abbreviation AS t2,
              player3_id AS p3, player3_name AS n3, player3_team_abbreviation AS t3
       FROM events WHERE eventmsgtype IN (1, 2, 3, 4, 5, 6, 7, 8, 10)""", con)
poss = pd.read_sql_query(
    "SELECT game_id, poss_num, period, seconds_left, is_home, home, away FROM poss ORDER BY game_id, poss_num", con)

# player names (for the leaderboards)
names = pd.concat([ev[[f"p{k}", f"n{k}"]].set_axis(["player_id", "player"], axis=1) for k in (1, 2, 3)])
names = names[(names["player_id"] > 0) & (names["player_id"] < 100_000_000) & names["player"].notna()]
names.drop_duplicates("player_id").astype({"player_id": int}).to_sql("players", con, if_exists="replace", index=False)

home_of = poss.groupby("game_id")["home"].first()
ev = ev[ev["game_id"].isin(home_of.index)]
ev["home"] = ev["game_id"].map(home_of)
technical = ev["txt"].str.contains("T.FOUL|T.Foul|TECH|Tech", regex=True) & (ev["t"] == 6)
ev = ev[~technical].sort_values(["game_id", "period", "seconds_left", "rid"], ascending=[True, True, False, True])

# for speed: plain Python lists, one tuple per play: (clock, is_sub, [(player, is_home), ...])
is_sub = (ev["t"] == 8).to_numpy()
clock = ev["seconds_left"].to_numpy()
people = []
for k in (1, 2, 3):
    pid = ev[f"p{k}"].fillna(0).to_numpy().astype("int64")
    team = ev[f"t{k}"].to_numpy()
    ok = (pid > 0) & (pid < 100_000_000) & pd.notna(team)          # real players only (team ids are huge numbers)
    side = (team == ev["home"].to_numpy())
    people.append((np.where(ok, pid, 0).tolist(), side.tolist()))
group = (ev["game_id"].to_numpy() * 10 + ev["period"].to_numpy())
starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
ends = np.r_[starts[1:], len(ev)]
ev_range = {int(group[s]): (s, e) for s, e in zip(starts, ends)}
is_sub, clock = is_sub.tolist(), clock.tolist()

pgroup = (poss["game_id"].to_numpy() * 10 + poss["period"].to_numpy())
pstarts = np.flatnonzero(np.r_[True, pgroup[1:] != pgroup[:-1]])
pends = np.r_[pstarts[1:], len(poss)]
pclock = poss["seconds_left"].to_numpy().tolist()
p_home_ball = (poss["is_home"] == 1).to_numpy().tolist()

n = len(poss)
off = np.zeros((n, 5), dtype="int64")
de = np.zeros((n, 5), dtype="int64")
complete = np.zeros(n, dtype=bool)
filled = short = 0
prev_game, prev_end = None, None

for ps, pe in zip(pstarts, pends):
    g = int(pgroup[ps])
    if g // 10 != prev_game:
        prev_game, prev_end = g // 10, None
    if g not in ev_range:
        prev_end = None
        continue
    s, e = ev_range[g]
    # pass 1: starters
    on = (set(), set())            # index 0 = away, 1 = home
    starters = (set(), set())
    came_in = (set(), set())
    went_out = (set(), set())
    for i in range(s, e):
        if is_sub[i]:
            out_p, in_p, side = people[0][0][i], people[1][0][i], int(people[0][1][i])
            if out_p and out_p not in on[side] and out_p not in came_in[side] and out_p not in went_out[side]:
                starters[side].add(out_p)
            on[side].discard(out_p)
            went_out[side].add(out_p)
            if in_p:
                on[side].add(in_p)
                came_in[side].add(in_p)
        else:
            for k in (0, 1, 2):
                p = people[k][0][i]
                if p:
                    side = int(people[k][1][i])
                    if p not in on[side] and p not in came_in[side] and p not in went_out[side]:
                        starters[side].add(p)
                        on[side].add(p)
    # silent starters: fill from the players who finished the previous period
    for side in (0, 1):
        missing = 5 - len(starters[side])
        if missing > 0 and prev_end is not None:
            candidates = prev_end[side] - starters[side] - came_in[side]
            if len(candidates) == missing:
                starters[side].update(candidates)
                filled += 1
        if len(starters[side]) != 5:
            short += 1
    usable = len(starters[0]) == 5 and len(starters[1]) == 5
    # pass 2: replay the substitutions, hand each possession its ten players
    on = (set(starters[0]), set(starters[1]))
    i = s
    for q in range(ps, pe):
        while i < e and clock[i] >= pclock[q]:
            if is_sub[i]:
                side = int(people[0][1][i])
                on[side].discard(people[0][0][i])
                if people[1][0][i]:
                    on[side].add(people[1][0][i])
            i += 1
        if usable and len(on[0]) == 5 and len(on[1]) == 5:
            o, d = (on[1], on[0]) if p_home_ball[q] else (on[0], on[1])
            off[q], de[q] = sorted(o), sorted(d)
            complete[q] = True
    while i < e:                                   # finish the period to know who ended it
        if is_sub[i]:
            side = int(people[0][1][i])
            on[side].discard(people[0][0][i])
            if people[1][0][i]:
                on[side].add(people[1][0][i])
        i += 1
    prev_end = (set(on[0]), set(on[1])) if usable else None

out = pd.DataFrame({"game_id": poss["game_id"], "poss_num": poss["poss_num"],
                    **{f"o{k + 1}": off[:, k] for k in range(5)}, **{f"d{k + 1}": de[:, k] for k in range(5)},
                    "complete": complete.astype(int)})
out.to_sql("lineups", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_lineups ON lineups(game_id, poss_num)")
con.commit()

# ---------------------------------------------------------------- checks
print(f"lineups: {n:,} possessions; complete ten-player lineup for {complete.mean():.1%}")
print(f"  periods where a silent starter was filled from the previous period: {filled:,}; team-periods left short: {short:,}")
scorers = pd.read_sql_query(
    "SELECT game_id, poss_num, player1_id AS pid FROM events WHERE eventmsgtype = 1 AND poss_num IS NOT NULL", con)
con.close()
row = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(poss[["game_id", "poss_num"]]))
r = row.reindex(pd.MultiIndex.from_frame(scorers[["game_id", "poss_num"]].astype("int64"))).to_numpy()
keep = ~np.isnan(r)
r, pid = r[keep].astype(int), scorers["pid"].to_numpy()[keep]
ok = complete[r]
hit = (off[r[ok]] == pid[ok][:, None]).any(axis=1)
print(f"  check: the player who made the basket is one of our five on offense in {hit.mean():.2%} of made shots")
print("Next: python expected.py")
