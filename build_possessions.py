"""
NBA Momentum Study - Step 2: one clean row per possession.

Builds three tables in data/nba.db:

  games   one row per game: home, away, final score.
  poss    one row per possession: game_id, season, period, seconds_left (in
          the period), offense, defense, is_home, start_margin (offense minus
          defense), points (scored by the offense), def_points (rare: points
          the DEFENSE scored during it, i.e. technical free throws),
          start_type, poss_num (order within the game), garbage-time flag.
  events  one row per play-by-play event, with the poss_num it happened in
          (used later to find dunks, blocks, steals, timeouts and who made
          them).

What the real data looks like (checked after the first download):
  - `possessions` is NOT one row per possession. Each possession is repeated
    once per play inside it; the `description` column holds that play's text
    (the same text as the play-by-play). We use this twice: dropping
    duplicates gives one row per possession, and the description text lets us
    link every play-by-play event to its possession.
  - Rows are not in game order, so we sort by period and clock.

How points are worked out. pbpstats gives the score margin at the START of
each possession, from the offense's point of view. So:
  - next possession by the OTHER team:  net = -(next margin) - (this margin)
  - next possession by the SAME team:   net = (next margin) - (this margin)
  - last possession of the game:        net from the final score
"net" is offense points minus anything the defense scored in that possession
(a technical free throw). We find those defense points from the linked
play-by-play events and add them back:  points = net + def_points.

Double check: `points_ok` is True when that answer equals a second,
independent count - simply adding up the offense's made shots and free throws
linked to the possession. validate.py also rebuilds every final score.

Usage:  python build_possessions.py [--db data/fake_nba.db]
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

# default database, or e.g.  --db data/fake_nba.db  for the test data
DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "data/nba.db"
GARBAGE_MARGIN = 20      # 4th quarter / OT, lead of 20+ = garbage time

con = sqlite3.connect(DB)


def need(df, cols, table):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{table} is missing columns {missing}.\n"
                         f"Available: {sorted(df.columns)}\n"
                         "The dataset's column names may have changed - adjust here.")


def to_seconds(v):
    """Handles '7:32', '07:32.0', 'PT07M32.00S' or plain numbers."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.startswith("PT"):
        m = s[2:].split("M")
        return float(m[0]) * 60 + float(m[1].rstrip("S") or 0)
    if ":" in s:
        mm, ss = s.split(":")[-2:]
        return float(mm) * 60 + float(ss)
    try:
        return float(s)
    except ValueError:
        return np.nan


def clock_seconds(col):
    """to_seconds for a whole column (fast: converts each distinct value once)."""
    return col.map({v: to_seconds(v) for v in col.unique()})


# ------------------------------------------------------------ play-by-play
pbp = pd.read_sql_query("SELECT rowid AS pbp_id, * FROM pbp", con)
need(pbp, ["game_id", "eventnum", "eventmsgtype", "period", "pctimestring", "score",
           "homedescription", "visitordescription", "player1_team_abbreviation"], "pbp")
pbp["game_id"] = pd.to_numeric(pbp["game_id"], errors="coerce").astype("int64")
pbp["team"] = pbp["player1_team_abbreviation"]
pbp["seconds_left"] = clock_seconds(pbp["pctimestring"])

# home / away team of each game = the team most often named in each column
home = (pbp[pbp["homedescription"].notna() & pbp["team"].notna()]
        .groupby("game_id")["team"].agg(lambda s: s.mode().iat[0]).rename("home"))
away = (pbp[pbp["visitordescription"].notna() & pbp["team"].notna()]
        .groupby("game_id")["team"].agg(lambda s: s.mode().iat[0]).rename("away"))
games = pd.concat([home, away], axis=1).dropna()

# SCORE is "VISITOR - HOME" and only filled in on scoring plays.
sc = pbp["score"].str.split("-", expand=True)
pbp["away_score"] = pd.to_numeric(sc[0].str.strip(), errors="coerce")
pbp["home_score"] = pd.to_numeric(sc[1].str.strip(), errors="coerce")
# Final score = the highest score reached (scores only go up). Safer than
# "the last row", because event numbers are not always in time order.
games["away_final"] = pbp.groupby("game_id")["away_score"].max()
games["home_final"] = pbp.groupby("game_id")["home_score"].max()

# Scoring plays: made shots (type 1) and made free throws (type 3). The
# scorer's side is the description column that mentions "(N PTS)".
made_home = pbp["homedescription"].fillna("").str.contains("PTS)", regex=False)
made_away = pbp["visitordescription"].fillna("").str.contains("PTS)", regex=False)
is_score = pbp["eventmsgtype"].isin([1, 3]) & pbp["score"].notna() & (made_home | made_away)
pbp["score_side"] = np.where(~is_score, None, np.where(made_home, "home", "away"))
# Points on each scoring play = how much that team's score went up. We line
# up a team's scoring plays by its own running score (again avoiding any
# reliance on event order).
pbp["pts"] = 0.0
for side in ("home", "away"):
    s = pbp.loc[pbp["score_side"] == side, ["game_id", f"{side}_score"]]
    s = s.sort_values(["game_id", f"{side}_score"], kind="stable")
    pbp.loc[s.index, "pts"] = s.groupby("game_id")[f"{side}_score"].diff().fillna(s[f"{side}_score"])

# ------------------------------------------------------------- possessions
# skip the two bulky text columns we never use (video link, full event list)
raw_cols = [r[1] for r in con.execute("PRAGMA table_info(possessions)") if r[1] not in ("url", "events")]
raw = pd.read_sql_query(f"SELECT {', '.join(raw_cols)} FROM possessions", con)
need(raw, ["gameid", "period", "starttime", "opponent", "startscoredifferential"], "possessions")
raw = raw.rename(columns={"gameid": "game_id"})
raw["game_id"] = pd.to_numeric(raw["game_id"], errors="coerce").astype("int64")
has_desc = "description" in raw

# one row per possession: drop the per-play column, keep distinct rows
pos = raw.drop(columns=["description"] if has_desc else []).drop_duplicates()
pos = pos.join(games, on="game_id", how="inner")

pos["seconds_left"] = clock_seconds(pos["starttime"])
pos["end_seconds_left"] = clock_seconds(pos["endtime"]) if "endtime" in pos else np.nan
pos["defense"] = pos["opponent"]
pos["offense"] = np.where(pos["defense"] == pos["home"], pos["away"], pos["home"])
pos["is_home"] = (pos["offense"] == pos["home"]).astype(int)
pos["start_margin"] = pd.to_numeric(pos["startscoredifferential"], errors="coerce")

# Game order: by period, then clock counting down. If two possessions start
# on the same second, the one that ends first came first.
pos = pos.sort_values(["game_id", "period", "seconds_left", "end_seconds_left"],
                      ascending=[True, True, False, False], kind="stable")
pos["poss_num"] = pos.groupby("game_id").cumcount() + 1

# ------------------------------------- link each play to its possession
# A play's text appears in both tables, so match on (game, period, text).
# The same text can occur twice in a period ("MISS Smith 3PT Jump Shot"), so
# when there are several candidates keep the possession whose clock window
# contains the play (or is nearest to it).
def clean(text):
    """The two files don't always spell names the same way ("Morris Sr." vs
    "Morris", stray double spaces in 2017-19), so tidy both before matching."""
    text = text.str.strip().str.replace(r"\s+", " ", regex=True)           # double spaces
    return text.str.replace(r" (Jr\.|Sr\.|III|II|IV)(?=[ :]|$)", "", regex=True)


ev_cols = ["pbp_id", "game_id", "period", "seconds_left"]
long = pd.concat([
    pbp.loc[pbp[c].notna(), ev_cols].assign(desc=clean(pbp.loc[pbp[c].notna(), c]))
    for c in ("homedescription", "visitordescription")])
# Plays involving both teams (block + missed shot, steal + turnover) are
# written as one line "home text: visitor text" in the possessions file.
both = pbp[pbp["homedescription"].notna() & pbp["visitordescription"].notna()]
long = pd.concat([long, both[ev_cols].assign(
    desc=clean(both["homedescription"]) + ": " + clean(both["visitordescription"]))])
if has_desc:
    key = ["game_id", "period", "starttime", "endtime", "opponent"]
    cand = raw[key + ["description"]].dropna(subset=["description"])
    cand = cand.assign(desc=clean(cand["description"])).drop(columns="description")
    cand = cand.merge(pos[key + ["poss_num", "seconds_left", "end_seconds_left"]]
                      .rename(columns={"seconds_left": "p_start", "end_seconds_left": "p_end"}),
                      on=key).drop_duplicates(["game_id", "period", "desc", "poss_num"])
    link = long.merge(cand[["game_id", "period", "desc", "poss_num", "p_start", "p_end"]],
                      on=["game_id", "period", "desc"])
    t = link["seconds_left"]
    link["dist"] = np.where((t <= link["p_start"]) & (t >= link["p_end"]), 0,
                            np.minimum((t - link["p_start"]).abs(), (t - link["p_end"]).abs()))
    link = link.sort_values(["pbp_id", "dist", "poss_num"]).drop_duplicates("pbp_id")
    pbp = pbp.merge(link[["pbp_id", "poss_num"]], on="pbp_id", how="left")
else:
    pbp["poss_num"] = np.nan

# points by each side inside each possession, from the linked scoring plays
scoring = pbp[pbp["score_side"].notna() & pbp["poss_num"].notna()]
by_side = (scoring.groupby(["game_id", "poss_num", "score_side"])["pts"].sum()
           .unstack(fill_value=0).reindex(columns=["home", "away"], fill_value=0))
pos = pos.join(by_side.rename(columns={"home": "ev_home", "away": "ev_away"}),
               on=["game_id", "poss_num"])
pos[["ev_home", "ev_away"]] = pos[["ev_home", "ev_away"]].fillna(0)
ev_off = np.where(pos["is_home"] == 1, pos["ev_home"], pos["ev_away"])
pos["def_points"] = np.where(pos["is_home"] == 1, pos["ev_away"], pos["ev_home"])

# net points from consecutive start margins (see docstring)
g = pos.groupby("game_id")
nxt_margin, nxt_off = g["start_margin"].shift(-1), g["offense"].shift(-1)
net = np.where(nxt_off == pos["offense"], nxt_margin - pos["start_margin"],
               -nxt_margin - pos["start_margin"])
final_margin = np.where(pos["is_home"] == 1, pos["home_final"] - pos["away_final"],
                        pos["away_final"] - pos["home_final"])
net = np.where(nxt_margin.isna(), final_margin - pos["start_margin"], net)
pos["points"] = net + pos["def_points"]
# trust a possession only when the two independent counts agree
pos["points_ok"] = (pos["points"] == ev_off) if has_desc else pos["points"].between(0, 6)

pos["garbage"] = ((pos["period"] >= 4) & (pos["start_margin"].abs() >= GARBAGE_MARGIN)).astype(int)

# ------------------------------------------------------------------ save
keep = ["game_id", "season", "playoffs", "gamedate", "period", "seconds_left", "end_seconds_left",
        "offense", "defense", "home", "away", "is_home", "start_margin", "points",
        "def_points", "points_ok", "starttype", "poss_num", "garbage",
        "fg2a", "fg2m", "fg3a", "fg3m", "turnovers", "offensiverebounds"]
out = pos[[c for c in keep if c in pos]]
out.to_sql("poss", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_poss_game ON poss(game_id, poss_num)")
games.reset_index(names="game_id").to_sql("games", con, if_exists="replace", index=False)

ev_keep = ["game_id", "eventnum", "poss_num", "period", "seconds_left", "eventmsgtype",
           "eventmsgactiontype", "homedescription", "visitordescription", "neutraldescription",
           "score_side", "pts", "player1_id", "player1_name", "player1_team_abbreviation",
           "player2_id", "player2_name", "player2_team_abbreviation",
           "player3_id", "player3_name", "player3_team_abbreviation"]
events = pbp[[c for c in ev_keep if c in pbp]]
events.to_sql("events", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_events_game ON events(game_id, poss_num)")
con.commit()
con.close()

print(f"poss: {len(out):,} possessions in {out['game_id'].nunique():,} games")
print(f"plays with text linked to a possession: {link['pbp_id'].nunique() / long['pbp_id'].nunique():.1%}"
      if has_desc else "no description column - plays not linked")
print(f"points agree with linked scoring plays for {out['points_ok'].mean():.1%} of possessions")
print(f"average points per possession: {out.loc[out['points_ok'], 'points'].mean():.3f}  (NBA is ~1.1)")
print("Next: python validate.py")
