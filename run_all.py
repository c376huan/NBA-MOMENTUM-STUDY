"""
NBA Momentum Study - run every step in order, with one command.

    python run_all.py                      # the real data in data/nba.db (downloads it first if needed)
    python run_all.py --from expected      # start part-way; the tables made by earlier steps must exist
    python run_all.py --fake 600           # a small simulated league in data/fake_nba.db: checks that
                                           # every step runs, start to finish, in a few minutes
    python run_all.py --db data/other.db   # any other database

Steps (each one reads the tables made by the ones before it):
    download_data  build_possessions  validate  download_coaches  build_lineups
    expected  momentum  players  coaches  pushes
With --fake the two download steps are skipped (the simulated league comes with its coaches).

Every step's printout is shown and also saved to logs/<step>.txt next to the database (inside data/,
so it is never committed). The run stops at the first step that fails.

Rough times for the real data (ten seasons) on a recent laptop: building the tables about 5 minutes,
expected.py 1 minute, momentum.py 5, players.py 2, coaches.py 1, pushes.py 2. The downloads need about
3 GB of disk for the raw files and 4 GB for the database.
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = ["download_data", "build_possessions", "validate", "download_coaches", "build_lineups",
         "expected", "momentum", "players", "coaches", "pushes"]
HERE = Path(__file__).resolve().parent


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    fake = arg("--fake", 0)
    db = arg("--db", "data/fake_nba.db" if fake else "data/nba.db")
    start = arg("--from", "build_possessions" if fake else STEPS[0])
    if start not in STEPS:
        raise SystemExit(f"--from must be one of: {', '.join(STEPS)}")
    steps = STEPS[STEPS.index(start):]
    if fake:
        steps = [s for s in steps if not s.startswith("download")]
    logs = Path(db).parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = [sys.executable, "-I"]

    def run(name, command):
        print(f"\n===== {name}  ({' '.join(command[2:])})", flush=True)
        began = time.time()
        with open(logs / f"{name}.txt", "w") as log:
            proc = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            proc.wait()
        print(f"===== {name}: {'done' if proc.returncode == 0 else 'FAILED'} in {time.time() - began:.0f} s", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"Stopped: {name} failed (its printout is in {logs / (name + '.txt')}).")

    if fake:
        run("make_fake_nba", python + [str(HERE / "tests" / "make_fake_nba.py"), "--games", str(fake), "--db", db])
    for name in steps:
        run(name, python + [str(HERE / f"{name}.py"), "--db", db])
    print(f"\nAll steps finished. Printouts are in {logs}/")


if __name__ == "__main__":
    main()
