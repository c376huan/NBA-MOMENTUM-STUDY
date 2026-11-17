"""Fake NBA database in the same shape as the real download, for testing the
pipeline and - most importantly - checking that momentum.py CAN find an
effect when one exists, and finds nothing when none exists.

The fake league has NO momentum unless you plant some:

    python tests/make_fake_nba.py                  # no momentum (expect ~0)
    python tests/make_fake_nba.py --momentum 0.2   # after a dunk, the dunking
                                                   # team scores +0.2 points per
                                                   # possession on its next 3
                                                   # possessions

Writes data/fake_nba.db (never touches the real data/nba.db). Then run:

    python build_possessions.py --db data/fake_nba.db
    python validate.py          --db data/fake_nba.db
    python momentum.py          --db data/fake_nba.db

Expected with --momentum 0.2: the "dunk" rows are clearly positive - about
+13 (points per 100 possessions) for the next 4 and 6 possessions and about
+7 for the next 10 (only 3 of the team's 5 possessions are boosted). That is
less than the full +20 planted, for two understandable reasons: dunks are so
common here that teams are already "boosted" ~1/6 of the time, and
momentum.py compares against a typical basket at the rim, which includes
dunks. The "8-0 run" rows also turn a little positive (runs contain dunks).
Blocks, steals, and-ones, 3-pointers and timeouts stay ~0.
Expected with no --momentum: every row ~0, and real runs = shuffled runs.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


MOMENTUM = arg("--momentum", 0.0)     # planted boost in points per possession
N_GAMES = arg("--games", 600)
DB = arg("--db", "data/fake_nba.db")

rng = np.random.default_rng(0)
TEAMS = ["TOR", "BOS", "NYK", "MIL", "DEN", "LAL", "PHX", "MIA", "DAL", "OKC"]
STRENGTH = dict(zip(TEAMS, rng.normal(0, 0.05, len(TEAMS))))   # points per possession
PLAYERS = {t: [(1000 * (i + 1) + k, f"{t.title()}{k}") for k in range(8)]
           for i, t in enumerate(TEAMS)}


def clock(t, pad):
    t = max(int(round(t)), 0)
    return f"{t // 60:02d}:{t % 60:02d}" if pad else f"{t // 60}:{t % 60:02d}"


pbp_rows, pos_rows = [], []
for gi in range(N_GAMES):
    gid = 22400001 + gi
    home, away = rng.choice(TEAMS, 2, replace=False)
    score = {home: 0, away: 0}
    stat = {}                                  # running player counts, for the text
    boost = {home: 0, away: 0}                 # boosted possessions left (planted momentum)
    run_team, run_pts = None, 0                # current unanswered run
    ev = 0

    def count(pid, what, add=1):
        stat[(pid, what)] = stat.get((pid, what), 0) + add
        return stat[(pid, what)]

    for period in (1, 2, 3, 4):
        t = 720.0
        off = home if (gi + period) % 2 else away
        start_type = "Off Dead Ball"
        while t > 0:
            de = away if off == home else home
            end = max(t - rng.uniform(8, 22), 0.0)
            if end < 5:
                end = 0.0
            margin = score[off] - score[de]
            plays = []          # (msgtype, action, off text, def text, p1, p2, p3, points)
            (pid, name), (did, dname) = (PLAYERS[off][rng.integers(8)], PLAYERS[de][rng.integers(8)])

            # chance of a made 2-pointer: base + team strength + home edge + planted momentum
            p_make2 = 0.29 + (STRENGTH[off] - STRENGTH[de]) / 2 + (0.005 if off == home else -0.005)
            if boost[off] > 0:
                p_make2 += MOMENTUM / 2
                boost[off] -= 1
            p = [0.13, p_make2, 0.125, 0.025, 0.06]            # turnover, 2, 3, and-one, 2 FTs
            kind = rng.choice(["to", "2", "3", "and1", "ft", "miss"], p=p + [1 - sum(p)])
            fg = dict(FG2A=0, FG2M=0, FG3A=0, FG3M=0, TURNOVERS=0)

            if kind == "to":
                fg["TURNOVERS"] = 1
                n = count(pid, "to")
                if rng.random() < 0.6:
                    plays.append((5, 1, f"{name} Bad Pass Turnover (P{n}.T{n})",
                                  f"{dname} STEAL ({count(did, 'stl')} STL)", (pid, name, off), (did, dname, de), None, 0))
                    next_start = "Off Steal"
                else:
                    plays.append((5, 4, f"{name} Traveling Turnover (P{n}.T{n})", None, (pid, name, off), None, None, 0))
                    next_start = "Off Dead Ball"
            elif kind in ("2", "and1"):
                fg["FG2A"] = fg["FG2M"] = 1
                dunk = kind == "2" and rng.random() < 0.2
                shot = "1' Dunk" if dunk else "2' Driving Layup"
                plays.append((1, 7 if dunk else 6, f"{name} {shot} ({count(pid, 'pts', 2)} PTS)", None,
                              (pid, name, off), None, None, 2))
                next_start = "Off At Rim Make"
                if dunk and MOMENTUM:
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
                plays.append((4, 0, None, f"{dname} REBOUND (Off:0 Def:1)", (did, dname, de), None, None, 0))

            # write the plays to both tables, in the real files' formats
            texts = []
            for i, (mtype, action, off_txt, def_txt, p1, p2, p3, pts) in enumerate(plays):
                ev += 1
                score[off] += pts
                home_txt, away_txt = (off_txt, def_txt) if off == home else (def_txt, off_txt)
                p1, p2, p3 = (x or (None, None, None) for x in (p1, p2, p3))
                when = end + (len(plays) - 1 - i > 0)       # last play at the end time
                pbp_rows.append(dict(
                    GAME_ID=gid, EVENTNUM=ev, EVENTMSGTYPE=mtype, EVENTMSGACTIONTYPE=action,
                    PERIOD=period, PCTIMESTRING=clock(when, False),
                    HOMEDESCRIPTION=home_txt, NEUTRALDESCRIPTION=None, VISITORDESCRIPTION=away_txt,
                    SCORE=f"{score[away]} - {score[home]}" if pts else None,
                    PLAYER1_ID=p1[0], PLAYER1_NAME=p1[1], PLAYER1_TEAM_ABBREVIATION=p1[2],
                    PLAYER2_ID=p2[0], PLAYER2_NAME=p2[1], PLAYER2_TEAM_ABBREVIATION=p2[2],
                    PLAYER3_ID=p3[0], PLAYER3_NAME=p3[1], PLAYER3_TEAM_ABBREVIATION=p3[2]))
                texts.append(f"{home_txt}: {away_txt}" if home_txt and away_txt else (home_txt or away_txt))
            for txt in texts:
                pos_rows.append(dict(
                    ENDTIME=clock(end, True), EVENTS="\n".join(texts) + "\n", **fg,
                    GAMEDATE="2024-11-01", GAMEID=gid, OFFENSIVEREBOUNDS=0, OPPONENT=de, PERIOD=period,
                    STARTSCOREDIFFERENTIAL=margin, STARTTIME=clock(t, True), STARTTYPE=start_type,
                    DESCRIPTION=txt, URL=""))

            # unanswered run so far; teams call more timeouts when being run on,
            # but in this fake world a timeout changes nothing
            got = sum(x[-1] for x in plays)
            if got:
                run_team, run_pts = (off, run_pts + got) if run_team == off else (off, got)
            start_type = next_start
            if end > 0 and rng.random() < (0.30 if run_pts >= 6 else 0.03):
                caller = (away if run_team == home else home) if run_pts >= 6 else rng.choice([home, away])
                ev += 1
                txt = f"{caller} Timeout: Regular (Full 1 Short 0)"
                pbp_rows.append(dict(
                    GAME_ID=gid, EVENTNUM=ev, EVENTMSGTYPE=9, EVENTMSGACTIONTYPE=1, PERIOD=period,
                    PCTIMESTRING=clock(end, False), HOMEDESCRIPTION=txt if caller == home else None,
                    VISITORDESCRIPTION=txt if caller == away else None))
                start_type = "Off Timeout"
            t, off = end, de

Path(DB).parent.mkdir(exist_ok=True)
con = sqlite3.connect(DB)
for table, rows in (("pbp", pbp_rows), ("possessions", pos_rows)):
    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]
    df["season"] = 2024
    df.to_sql(table, con, if_exists="replace", index=False)
con.close()
print(f"fake db -> {DB}: {N_GAMES} games, planted momentum = {MOMENTUM} points per possession")
