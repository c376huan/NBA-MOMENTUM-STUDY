"""
NBA Momentum Study - Step 3: check the data before trusting it.

1. Rebuilds every game's final score by adding up possession points and
   compares it with the official final score from the play-by-play.
2. Basic sanity numbers (possessions per game, points per possession).
3. Prints 5 random games to check by hand against NBA.com box scores.

Usage:  python validate.py
"""

import sqlite3
import sys

import pandas as pd

DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "data/nba.db"
con = sqlite3.connect(DB)
poss = pd.read_sql_query("SELECT * FROM poss", con)
games = pd.read_sql_query("SELECT * FROM games", con).set_index("game_id")

# A team's total = points on its own possessions + the (rare) points it
# scored while defending (technical free throws).
pts = poss.groupby(["game_id", "is_home"])["points"].sum().unstack()
dpts = poss.groupby(["game_id", "is_home"])["def_points"].sum().unstack()
check = games.join(pd.DataFrame({"home_rebuilt": pts[1] + dpts[0],
                                 "away_rebuilt": pts[0] + dpts[1]}), how="inner")
check["ok"] = ((check["home_rebuilt"] == check["home_final"])
               & (check["away_rebuilt"] == check["away_final"]))

print(f"Games checked: {len(check):,}")
print(f"Final score rebuilt exactly from possessions: {check['ok'].mean():.1%}")
per_game = poss.groupby("game_id").size()
print(f"Possessions per game: median {per_game.median():.0f} (expect ~195-205 incl. both teams)")
valid = poss[poss["points_ok"].astype(bool)]
print(f"Points per possession: {valid['points'].mean():.3f} (expect ~1.10-1.16)")
print(f"Garbage-time possessions flagged: {poss['garbage'].mean():.1%}")

bad = check[~check["ok"]]
if len(bad):
    print(f"\n{len(bad)} games don't rebuild exactly - first few:")
    print(bad[["home", "away", "home_final", "home_rebuilt", "away_final", "away_rebuilt"]].head())
    print("A few mismatches are normal (data glitches); many = a bug in build_possessions.py.")

print("\nSpot-check these on NBA.com (search the teams + date):")
sample = check.sample(min(5, len(check)), random_state=1)
dates = poss.drop_duplicates("game_id").set_index("game_id").get("gamedate")
for gid, r in sample.iterrows():
    d = dates.get(gid, "?") if dates is not None else "?"
    print(f"  {d}  {r['away']} {int(r['away_final'])} @ {r['home']} {int(r['home_final'])}")
