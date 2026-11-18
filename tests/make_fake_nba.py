"""Fake NBA database in the same shape as the real download, for checking that
every step of the analysis CAN find an effect when one exists and finds
nothing when none exists.

The fake league: 10 teams, 9 players each, 5 on the floor, real-looking
substitutions (including silent lineup changes between periods), players of
different quality on offense and defense, the possession chain (a possession
after a steal is worth more, after a timeout a little less), one coach per
team, timeouts that are called more often when momentum is against a team.
By default NOTHING else depends on momentum. Plant effects with:

  --momentum 0.2    after ANY dunk the dunking team scores +0.2 points per
                    possession on its next 3 trips
  --creators        ... only after dunks by players whose name ends in 0 or 1
  --stoppers 0.3    players whose name ends in 2 lift their team's offense
                    when momentum is against it (up to +0.3 points per
                    possession when momentum is at -100 or worse)
  --coaches 0.2     after a timeout by the coaches of the first two teams
                    their team scores +0.2 points per possession on its next
                    3 trips; for the last two teams it is -0.2
  --stars           players whose name ends in 4 take three times as many of
                    their team's plays (they should top STARTS and LEADS in
                    pushes.py)
  --closers         players whose name ends in 3 take more of their team's
                    plays the more momentum is against it (they should top
                    STOPS in pushes.py, but not LEADS)
  --igniters        players whose name ends in 5 take more of their team's
                    plays the more momentum is WITH it (they should top the
                    "more than his usual" LEADS list in pushes.py; the stars
                    should not, because a star does the same all the time)

    python tests/make_fake_nba.py --games 3000 --stoppers 0.3 --coaches 0.2

Writes data/fake_nba.db (never touches the real data/nba.db). Then run each
step with  --db data/fake_nba.db :
    build_possessions.py, validate.py, build_lineups.py, expected.py,
    momentum.py, players.py, coaches.py

What to expect: with nothing planted every momentum result is ~0 (player
QUALITY is still found - that is real in the fake league). Each planted
effect should show up in the matching place and nowhere else: dunks in
momentum.py part D and players.py "after his plays"; the "...2" players at
the top of players.py "when momentum is against"; TOR/BOS coaches at the top
and DAL/OKC at the bottom of coaches.py and pushes.py; the "...4" and "...3"
players in pushes.py as described above.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


MOMENTUM = arg("--momentum", 0.0)
ONLY_CREATORS = "--creators" in sys.argv
STOPPERS = arg("--stoppers", 0.0)
COACHES = arg("--coaches", 0.0)
STARS = "--stars" in sys.argv             # players ending in 4 take three times as many of their team's plays
CLOSERS = "--closers" in sys.argv         # players ending in 3 take more of the plays the more momentum is against
IGNITERS = "--igniters" in sys.argv       # players ending in 5 take more of the plays the more momentum is with their team
N_GAMES = arg("--games", 600)
DB = arg("--db", "data/fake_nba.db")
KEEP = 0.5 ** (1 / 4)                    # the generator's own momentum tracker (half-life 4)

rng = np.random.default_rng(0)
TEAMS = ["TOR", "BOS", "NYK", "MIL", "DEN", "LAL", "PHX", "MIA", "DAL", "OKC"]
PLAYERS = {t: [(1000 * (i + 1) + k, f"{t.title()}{k}") for k in range(9)] for i, t in enumerate(TEAMS)}
OFF = {pid: v for t in TEAMS for (pid, _), v in zip(PLAYERS[t], rng.normal(0, 0.02, 9))}    # points per possession
DEF = {pid: v for t in TEAMS for (pid, _), v in zip(PLAYERS[t], rng.normal(0, 0.02, 9))}    # allowed (negative = good)
# the possession chain, as in real games: how a possession starts changes what it is worth
CHAIN = {"Off Steal": 0.20, "Off At Rim Block": 0.08, "Off At Rim Miss": 0.05, "Off Arc 3 Miss": 0.03, "Off Timeout": -0.03}
COACH_EFFECT = {t: (COACHES if i < 2 else -COACHES if i >= 8 else 0.0) for i, t in enumerate(TEAMS)}


def clock(t, pad):
    t = max(int(round(t)), 0)
    return f"{t // 60:02d}:{t % 60:02d}" if pad else f"{t // 60}:{t % 60:02d}"


pbp_rows, pos_rows, coach_rows = [], [], []
ev = 0
for gi in range(N_GAMES):
    gid = 22400001 + gi
    home, away = rng.choice(TEAMS, 2, replace=False)
    coach_rows += [(gid, home, f"Coach {home}"), (gid, away, f"Coach {away}")]
    score = {home: 0, away: 0}
    stat = {}
    boost = {home: 0, away: 0}                 # trips left with planted dunk momentum
    to_boost = {home: 0, away: 0}              # trips left with a planted timeout effect
    rate = {home: 0.0, away: 0.0}              # each team's recent offense rate (points per 100 vs average)
    on = {t: list(rng.choice(9, 5, replace=False)) for t in (home, away)}

    def count(pid, what, add=1):
        stat[(pid, what)] = stat.get((pid, what), 0) + add
        return stat[(pid, what)]

    def row(mtype, action, period, when, home_txt, away_txt, p1=None, p2=None, p3=None, pts_made=False):
        global ev
        ev += 1
        p1, p2, p3 = (x or (None, None, None) for x in (p1, p2, p3))
        pbp_rows.append(dict(
            GAME_ID=gid, EVENTNUM=ev, EVENTMSGTYPE=mtype, EVENTMSGACTIONTYPE=action, PERIOD=period,
            PCTIMESTRING=clock(when, False), HOMEDESCRIPTION=home_txt, NEUTRALDESCRIPTION=None,
            VISITORDESCRIPTION=away_txt, SCORE=f"{score[away]} - {score[home]}" if pts_made else None,
            PLAYER1_ID=p1[0], PLAYER1_NAME=p1[1], PLAYER1_TEAM_ABBREVIATION=p1[2],
            PLAYER2_ID=p2[0], PLAYER2_NAME=p2[1], PLAYER2_TEAM_ABBREVIATION=p2[2],
            PLAYER3_ID=p3[0], PLAYER3_NAME=p3[1], PLAYER3_TEAM_ABBREVIATION=p3[2]))

    def substitute(team, period, when):
        out_k = on[team][rng.integers(5)]
        in_k = rng.choice([k for k in range(9) if k not in on[team]])
        on[team][on[team].index(out_k)] = in_k
        (po, no), (pi, ni) = PLAYERS[team][out_k], PLAYERS[team][in_k]
        txt = f"SUB: {ni} FOR {no}"
        row(8, 0, period, when, txt if team == home else None, txt if team == away else None,
            (po, no, team), (pi, ni, team))

    for period in (1, 2, 3, 4):
        t = 720.0
        off = home if (gi + period) % 2 else away
        start_type = "Off Dead Ball"
        if period > 1:                           # silent lineup changes between periods (as in real data)
            for team in (home, away):
                for _ in range(rng.integers(0, 3)):
                    out_k = on[team][rng.integers(5)]
                    on[team][on[team].index(out_k)] = rng.choice([k for k in range(9) if k not in on[team]])
        while t > 0:
            de = away if off == home else home
            end = max(t - rng.uniform(8, 22), 0.0)
            if end < 5:
                end = 0.0
            margin = score[off] - score[de]
            plays = []          # (msgtype, action, offense text, defense text, p1, p2, p3, points)
            share = np.ones(5)                                   # who takes the play on offense
            if STARS:
                share[[k == 4 for k in on[off]]] *= 3
            if CLOSERS:
                share[[k == 3 for k in on[off]]] *= 1 + 4 * min(max(-(rate[off] - rate[de]), 0.0), 100.0) / 100.0
            if IGNITERS:
                share[[k == 5 for k in on[off]]] *= 1 + 4 * min(max(rate[off] - rate[de], 0.0), 100.0) / 100.0
            k_off, k_def = on[off][rng.choice(5, p=share / share.sum())], on[de][rng.integers(5)]
            (pid, name), (did, dname) = PLAYERS[off][k_off], PLAYERS[de][k_def]
            mate = PLAYERS[off][[k for k in on[off] if k != k_off][rng.integers(4)]]

            # points per possession above average for this possession
            edge = sum(OFF[PLAYERS[off][k][0]] for k in on[off]) + sum(DEF[PLAYERS[de][k][0]] for k in on[de])
            edge += 0.01 if off == home else -0.01
            edge += CHAIN.get(start_type, 0.0)
            if boost[off] > 0:
                edge += MOMENTUM
                boost[off] -= 1
            if to_boost[off] > 0:
                edge += COACH_EFFECT[off]
                to_boost[off] -= 1
            against = max(-(rate[off] - rate[de]), 0.0)          # momentum against the team with the ball
            if STOPPERS and any(k == 2 for k in on[off]):
                edge += STOPPERS * min(against, 100.0) / 100.0
            p_make2 = min(max(0.29 + edge / 2, 0.02), 0.60)     # keep it a valid probability
            p = [0.13, p_make2, 0.125, 0.025, 0.06]            # turnover, 2, 3, and-one, 2 FTs
            kind = rng.choice(["to", "2", "3", "and1", "ft", "miss"], p=p + [1 - sum(p)])
            fg = dict(FG2A=0, FG2M=0, FG3A=0, FG3M=0, TURNOVERS=0)

            if kind == "to":
                fg["TURNOVERS"] = 1
                n_ = count(pid, "to")
                if rng.random() < 0.6:
                    plays.append((5, 1, f"{name} Bad Pass Turnover (P{n_}.T{n_})",
                                  f"{dname} STEAL ({count(did, 'stl')} STL)", (pid, name, off), (did, dname, de), None, 0))
                    next_start = "Off Steal"
                else:
                    plays.append((5, 4, f"{name} Traveling Turnover (P{n_}.T{n_})", None, (pid, name, off), None, None, 0))
                    next_start = "Off Dead Ball"
            elif kind in ("2", "and1"):
                fg["FG2A"] = fg["FG2M"] = 1
                dunk = kind == "2" and rng.random() < 0.2
                shot = "1' Dunk" if dunk else "2' Driving Layup"
                assist = f" ({mate[1]} {count(mate[0], 'ast')} AST)" if rng.random() < 0.5 else ""
                plays.append((1, 7 if dunk else 6, f"{name} {shot} ({count(pid, 'pts', 2)} PTS){assist}", None,
                              (pid, name, off), (mate[0], mate[1], off) if assist else None, None, 2))
                next_start = "Off At Rim Make"
                if dunk and MOMENTUM and (not ONLY_CREATORS or pid % 1000 < 2):
                    boost[off] = 3
                if kind == "and1":
                    plays.append((6, 2, None, f"{dname} S.FOUL (P{count(did, 'pf')}.T1)", (did, dname, de), (pid, name, off), None, 0))
                    if rng.random() < 0.75:
                        plays.append((3, 10, f"{name} Free Throw 1 of 1 ({count(pid, 'pts', 1)} PTS)", None, (pid, name, off), None, None, 1))
                        next_start = "Off FT Make"
                    else:
                        plays.append((3, 10, f"MISS {name} Free Throw 1 of 1", None, (pid, name, off), None, None, 0))
                        next_start = "Off FT Miss"
            elif kind == "3":
                fg["FG3A"] = fg["FG3M"] = 1
                plays.append((1, 1, f"{name} 26' 3PT Jump Shot ({count(pid, 'pts', 3)} PTS)", None, (pid, name, off), None, None, 3))
                next_start = "Off Arc 3 Make"
            elif kind == "ft":
                plays.append((6, 2, None, f"{dname} S.FOUL (P{count(did, 'pf')}.T1)", (did, dname, de), (pid, name, off), None, 0))
                for k in (1, 2):
                    if rng.random() < 0.77:
                        plays.append((3, 10 + k, f"{name} Free Throw {k} of 2 ({count(pid, 'pts', 1)} PTS)", None, (pid, name, off), None, None, 1))
                        next_start = "Off FT Make"
                    else:
                        plays.append((3, 10 + k, f"MISS {name} Free Throw {k} of 2", None, (pid, name, off), None, None, 0))
                        next_start = "Off FT Miss"
            else:
                three = rng.random() < 0.45
                fg["FG3A" if three else "FG2A"] = 1
                if not three and rng.random() < 0.25:
                    plays.append((2, 1, f"MISS {name} 2' Layup", f"{dname} BLOCK ({count(did, 'blk')} BLK)",
                                  (pid, name, off), None, (did, dname, de), 0))
                    next_start = "Off At Rim Block"
                else:
                    plays.append((2, 1, f"MISS {name} {'26\' 3PT Jump Shot' if three else '2\' Layup'}", None,
                                  (pid, name, off), None, None, 0))
                    next_start = "Off Arc 3 Miss" if three else "Off At Rim Miss"
                rk = on[de][rng.integers(5)]
                plays.append((4, 0, None, f"{PLAYERS[de][rk][1]} REBOUND (Off:0 Def:1)", (*PLAYERS[de][rk], de), None, None, 0))

            texts = []
            for i, (mtype, action, off_txt, def_txt, p1, p2, p3, pts) in enumerate(plays):
                score[off] += pts
                home_txt, away_txt = (off_txt, def_txt) if off == home else (def_txt, off_txt)
                row(mtype, action, period, end + (len(plays) - 1 - i > 0), home_txt, away_txt, p1, p2, p3, pts_made=bool(pts))
                texts.append(f"{home_txt}: {away_txt}" if home_txt and away_txt else (home_txt or away_txt))
            for txt in texts:
                pos_rows.append(dict(
                    ENDTIME=clock(end, True), EVENTS="", **fg, GAMEDATE="2024-11-01", GAMEID=gid, OFFENSIVEREBOUNDS=0,
                    OPPONENT=de, PERIOD=period, STARTSCOREDIFFERENTIAL=margin, STARTTIME=clock(t, True),
                    STARTTYPE=start_type, DESCRIPTION=txt, URL=""))

            got = sum(x[-1] for x in plays)
            rate[off] = KEEP * rate[off] + (1 - KEEP) * (got - 1.12) * 100
            start_type = next_start
            if end > 0:
                for team in (home, away):                       # substitutions at the stoppage
                    if rng.random() < 0.08:
                        substitute(team, period, end)
                # the team about to get the ball calls timeout more often when momentum is against it
                nxt = de
                m_against = max(-(rate[nxt] - rate[off]), 0.0)
                if rng.random() < 0.02 + 0.10 * min(m_against, 100.0) / 100.0:
                    txt = f"{nxt} Timeout: Regular (Full 1 Short 0)"
                    row(9, 1, period, end, txt if nxt == home else None, txt if nxt == away else None)
                    start_type = "Off Timeout"
                    if COACHES:
                        to_boost[nxt] = 3
            t, off = end, de

Path(DB).parent.mkdir(exist_ok=True)
con = sqlite3.connect(DB)
for table, rows in (("pbp", pbp_rows), ("possessions", pos_rows)):
    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]
    df["season"], df["playoffs"] = 2024, 0
    df.to_sql(table, con, if_exists="replace", index=False)
pd.DataFrame(coach_rows, columns=["game_id", "team", "coach"]).to_sql("game_coach", con, if_exists="replace", index=False)
con.close()
print(f"fake db -> {DB}: {N_GAMES} games; planted: dunk momentum {MOMENTUM}"
      f"{' (creators only)' if ONLY_CREATORS else ''}, stoppers {STOPPERS}, coaches {COACHES}")
