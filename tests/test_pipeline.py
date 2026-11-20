"""
End-to-end check on two small simulated leagues: does the pipeline find nothing when nothing is planted,
and find what is planted when it is?

    python tests/test_pipeline.py          (about 5 minutes; also runs under pytest)

League 1, nothing planted: carry-over beyond the possession chain must be indistinguishable from zero,
and players must not be told apart on momentum.
League 2, planted (tests/make_fake_nba.py --stoppers 0.3 --coaches 0.2): the ten "...2" players help
more when momentum is against their team, and the timeouts of two coaches help (TOR, BOS) while two
hurt (DAL, OKC). players.py must find the stoppers and coaches.py must give the coaches the right sign.

Each step's printout is checked with plain text matching, so if a printout changes, update the
patterns here. The leagues are built in a temporary folder and deleted afterwards.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GAMES = 2000


def run(script, db, *extra):
    out = subprocess.run([sys.executable, "-I", str(HERE / script), "--db", str(db), *extra],
                         cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0, f"{script} failed:\n{out.stderr[-2000:]}"
    return out.stdout


def build(db, *planted):
    run("tests/make_fake_nba.py", db, "--games", str(GAMES), *planted)
    for step in ("build_possessions.py", "build_lineups.py", "expected.py"):
        run(step, db)
    return run("momentum.py", db, "--shuffles", "10")


def interval_at(printout, layer, column):
    """(low, high) of the points-per-100 table for a layer, at the given half-life column (0 = half-life 1)."""
    table = printout[printout.index("The same, in points per 100"):]
    row = next(line for line in table.splitlines() if line.strip().startswith(layer))
    cells = re.findall(r"([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]", row)
    return float(cells[column][1]), float(cells[column][2])


def top_names(printout, title, k=10):
    """The first k player names listed under a title."""
    block = printout[printout.index(title):]
    lines = block.splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith("player"))
    return [re.split(r"\s{2,}", line.strip())[0] for line in lines[head + 1:head + 1 + k]]


def coach_effects(printout):
    """Effect on his team, per coach, from the regular-season coach table of coaches.py."""
    effects = {}
    for line in printout.splitlines():
        m = re.match(r"^\s*Coach ([A-Z]{3})\s+\d+\s+[\d,]+\s+[+-]?\d+\.\d\s+([+-]\d+\.\d)", line)
        if m:
            effects.setdefault(m.group(1), float(m.group(2)))
    return effects


def test_null_and_planted_leagues():
    folder = Path(tempfile.mkdtemp(prefix="nba_fake_"))
    try:
        # --- nothing planted
        null = folder / "null.db"
        momentum = build(null)
        for layer in ("chain removed", "chain + players on the floor removed"):
            lo, hi = interval_at(momentum, layer, 2)                     # half-life 4
            assert lo < 0 < hi, f"null league: {layer} carry-over interval {lo:+.2f} to {hi:+.2f} excludes 0"
        players = run("players.py", null)
        assert "Do players differ in how they respond to momentum?  no" in players, "null league: players told apart"

        # --- planted stoppers and coaches
        planted = folder / "planted.db"
        build(planted, "--stoppers", "0.3", "--coaches", "0.2")
        players = run("players.py", planted)
        assert "Do players differ in how they respond to momentum?  yes" in players, "planted league: stoppers not found"
        top = top_names(players, "STOPS IT - ")
        found = sum(name.endswith("2") for name in top)
        assert found >= 8, f"planted league: only {found} of the top 10 in STOPS IT are planted stoppers: {top}"
        effects = coach_effects(run("coaches.py", planted))
        for team in ("TOR", "BOS"):
            assert effects[team] > 0, f"planted league: coach {team} should help, got {effects[team]:+.1f}"
        for team in ("DAL", "OKC"):
            assert effects[team] < 0, f"planted league: coach {team} should hurt, got {effects[team]:+.1f}"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    test_null_and_planted_leagues()
    print("ok  the null league shows nothing and the planted league shows what was planted")
