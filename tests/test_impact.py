"""
Check impact.py against a simulated league in which every player's true quality is known.

    python tests/test_impact.py          (about 2 minutes; also runs under pytest)

The league has four seasons (tests/make_fake_nba.py --seasons 4): player quality drifts a little
between seasons and some players change teams. Both the old OVERALL IMPACT (expected.py's single
seasons, averaged by players.py) and impact.py's are compared with the truth, each measured against
the average player of the same season. impact.py must
  1. reproduce expected.py's single-season estimates when given the old settings,
  2. be closer to the truth than the old way, and
  3. give 95% intervals that contain the true value for most players (in three test leagues the new
     intervals did so for 90-92% of players, the old way's for only 78-82%).
The league is built in a temporary folder and deleted afterwards.
"""

import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def run(script, db, *extra):
    out = subprocess.run([sys.executable, "-I", str(HERE / script), "--db", str(db), *extra],
                         cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0, f"{script} failed:\n{out.stderr[-2000:]}"
    return out.stdout


def centre(d, col):
    """Measured against the season's average player, weighted by possessions."""
    pw = d["possessions"] * d[col]
    return d[col] - pw.groupby(d["season"]).transform("sum") / d["possessions"].groupby(d["season"]).transform("sum")


def scores(db):
    """Error against the truth and share of 95% intervals containing it, for the old way and impact.py."""
    con = sqlite3.connect(db)
    truth = pd.read_sql_query("SELECT season, player_id, off - def AS net_true FROM fake_truth", con)
    seasons = pd.read_sql_query("SELECT * FROM player_season_impact", con).merge(truth, on=["season", "player_id"])
    new = pd.read_sql_query("SELECT * FROM player_impact", con).set_index("player_id")
    old = pd.read_sql_query("SELECT * FROM player_effects", con)
    con.close()
    seasons["net_true"] = centre(seasons, "net_true")
    old = old.merge(seasons[["season", "player_id", "possessions", "net_true"]], on=["season", "player_id"])
    old["net"] = centre(old, "net")
    w = 1 / old["net_se"] ** 2                                   # players.py's old season average
    g = old.assign(w=w, vw=old["net"] * w, tw=old["net_true"] * w).groupby("player_id")
    old_avg = pd.DataFrame({"est": g["vw"].sum() / g["w"].sum(), "se": 1 / np.sqrt(g["w"].sum()),
                            "true": g["tw"].sum() / g["w"].sum()})
    gn = seasons.assign(tw=seasons["net_true"] * seasons["possessions"]).groupby("player_id")
    new_avg = pd.DataFrame({"est": new["impact"], "se": new["se"], "true": gn["tw"].sum() / gn["possessions"].sum()})
    out = {}
    for name, d in (("old", old_avg), ("new", new_avg)):
        e = d["est"] - d["true"]
        out[name] = (np.sqrt((e ** 2).mean()), (np.abs(e) <= 1.96 * d["se"]).mean())
    return out


def test_impact_against_truth():
    folder = Path(tempfile.mkdtemp(prefix="nba_impact_"))
    try:
        db = folder / "league.db"
        run("tests/make_fake_nba.py", db, "--games", "500", "--seasons", "4", "--seed", "3")
        for step in ("build_possessions.py", "build_lineups.py", "expected.py"):
            run(step, db)
        printout = run("impact.py", db)
        assert "reproduces expected.py's player effects" in printout and "WARNING" not in printout, \
            "impact.py did not reproduce expected.py's single-season estimates"
        s = scores(db)
        (old_err, old_cover), (new_err, new_cover) = s["old"], s["new"]
        print(f"error vs truth: old {old_err:.2f}, new {new_err:.2f}; 95% intervals containing the truth: "
              f"old {old_cover:.0%}, new {new_cover:.0%}")
        assert new_err < old_err, f"impact.py is not closer to the truth ({new_err:.2f} vs {old_err:.2f})"
        assert new_cover >= 0.85, f"impact.py's 95% intervals contain the truth for only {new_cover:.0%} of players"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    test_impact_against_truth()
    print("ok  impact.py is closer to the truth than the old way and its intervals are about right")
