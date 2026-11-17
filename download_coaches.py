"""
NBA Momentum Study - head coaches for every game.

The play-by-play data has no coach names, so this adds them. Source: the
season coach tables on Basketball-Reference.com (one small web page per
season: coach, team, games, wins, losses). Run build_possessions.py first.

When a team changed coach mid-season the page gives each coach's record but
not the dates. We work out the order ourselves: the coaches' win-loss
records must add up along the team's actual game results (from our `games`
table), and only the right order fits. Any team-season where nothing fits is
printed, and its games are left without a coach.

Safety: we only fetch from basketball-reference.com, one page every 4
seconds (their limit for scripts is 20 per minute), and the pages are only
read as text - nothing downloaded is run.

Playoff games get the coach who finished the regular season with the team.

Output: table `game_coach` (game_id, team, coach) in data/nba.db.
The pages are saved under data/raw/coaches/ so re-runs don't fetch again.

Usage:  python download_coaches.py
"""

import re
import sqlite3
import sys
import time
from itertools import permutations
from pathlib import Path

import pandas as pd
import requests

DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "data/nba.db"
URL = "https://www.basketball-reference.com/leagues/NBA_{year}_coaches.html"   # year = season END year
RAW = Path("data/raw/coaches")
MAX_BYTES = 5 * 1024**2
# Basketball-Reference team codes that differ from the NBA's
TEAM_FIX = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}


def page(season):
    """HTML of the coach table for a season (season = start year, e.g. 2024 for 2024-25)."""
    path = RAW / f"coaches_{season}.html"
    if not path.exists():
        print(f"  fetching {season} ...", flush=True)
        r = requests.get(URL.format(year=season + 1), timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (personal research script)"})
        r.raise_for_status()
        if len(r.content) > MAX_BYTES or b"<table" not in r.content:
            raise RuntimeError(f"{season}: page is not what we expected - stopping")
        path.write_bytes(r.content)
        time.sleep(4)
    return path.read_text(encoding="utf-8", errors="replace")


def parse(html, season):
    """Pull (coach, team, games, wins, losses) out of the table rows."""
    table = html[html.find("<table"):html.find("</table>")]
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", table, flags=re.S):
        cell = {k: re.sub(r"<[^>]+>", "", v).strip()
                for k, v in re.findall(r'<t[hd][^>]*data-stat="([^"]*)"[^>]*>(.*?)</t[hd]>', tr, flags=re.S)}
        if cell.get("cur_g", "").isdigit():
            rows.append(dict(season=season, coach=cell["coach"].replace("&nbsp;", " "),
                             team=TEAM_FIX.get(cell["team"], cell["team"]),
                             g=int(cell["cur_g"]), w=int(cell["cur_w"]), l=int(cell["cur_l"])))
    return rows


con = sqlite3.connect(DB)
games = pd.read_sql_query(
    """SELECT g.game_id, g.home, g.away, g.home_final, g.away_final, p.season, p.gamedate, p.playoffs
       FROM games g JOIN (SELECT game_id, MIN(season) AS season, MIN(gamedate) AS gamedate,
                                 MAX(playoffs) AS playoffs
                          FROM poss GROUP BY game_id) p USING (game_id)""", con)
RAW.mkdir(parents=True, exist_ok=True)
coaches = pd.DataFrame([row for s in sorted(games["season"].unique()) for row in parse(page(int(s)), int(s))])

# each team's games in the order they were played, with win = 1
home = games.assign(team=games["home"], win=(games["home_final"] > games["away_final"]).astype(int))
away = games.assign(team=games["away"], win=(games["away_final"] > games["home_final"]).astype(int))
all_team_games = pd.concat([home, away]).sort_values(["gamedate", "game_id"])[["season", "team", "game_id", "win", "playoffs"]]
team_games = all_team_games[all_team_games["playoffs"] == 0]          # the page's records are regular season

out, problems = [], []
for (season, team), tg in team_games.groupby(["season", "team"]):
    staff = coaches[(coaches["season"] == season) & (coaches["team"] == team)]
    wins = tg["win"].to_numpy()
    fit = (0,) if len(staff) == 1 else None            # one coach all season: nothing to work out
    for order in permutations(range(len(staff))) if fit is None else ():   # else try every order
        start, good = 0, True
        for k in order:
            g, w = staff["g"].iloc[k], staff["w"].iloc[k]
            good &= wins[start:start + g].sum() == w
            start += g
        # our data may miss a game or two, so allow the totals to be slightly short
        if good and abs(start - len(wins)) <= 2:
            fit = order
            break
    if fit is None:
        problems.append((season, team, list(staff["coach"]), len(wins), int(staff["g"].sum())))
        continue
    start = 0
    for k in fit:
        g = staff["g"].iloc[k]
        out += [(gid, team, staff["coach"].iloc[k]) for gid in tg["game_id"].iloc[start:start + g]]
        start += g

game_coach = pd.DataFrame(out, columns=["game_id", "team", "coach"])
# playoffs: the coach who finished the regular season with the team
last = (game_coach.merge(team_games[["game_id", "team", "season"]], on=["game_id", "team"])
        .groupby(["season", "team"])["coach"].last())
po = all_team_games[all_team_games["playoffs"] == 1]
po = po.assign(coach=last.reindex(pd.MultiIndex.from_frame(po[["season", "team"]])).to_numpy()).dropna(subset=["coach"])
game_coach = pd.concat([game_coach, po[["game_id", "team", "coach"]]])
game_coach.to_sql("game_coach", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_game_coach ON game_coach(game_id, team)")
con.commit()
con.close()

print(f"{len(coaches)} coach-team-seasons; coach assigned for {len(game_coach):,} of {len(all_team_games):,} team-games "
      f"({game_coach['coach'].nunique()} different head coaches)")
if problems:
    print(f"\n{len(problems)} team-seasons where the records did not fit (left without a coach):")
    for p in problems:
        print("  season %s %s coaches=%s  games in our data=%s, on the page=%s" % p)
print("Next: python coaches.py")
