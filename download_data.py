"""
NBA Momentum Study - Step 1: download play-by-play + possession data.

Source: the free shufinskiy/nba_data dataset on GitHub (Apache-2.0),
compiled from stats.nba.com ("nbastats") and pbpstats.com ("pbpstats").
Each file is a .tar.xz archive containing one CSV per season.

  nbastats_<year>  every play (who, what, score)          -> table pbp
  pbpstats_<year>  one row per play, grouped by           -> table possessions
                   possession (start/end, start type,
                   score margin, shots...)
  ..._po_<year>    the same for the playoffs (column playoffs = 1)

<year> is the season's START year (2024 = 2024-25). The dataset currently
ends at 2024-25; the default is the 10 seasons 2015-16 .. 2024-25.

Usage:
    pip install requests pandas
    python download_data.py                 # default: 10 seasons
    python download_data.py 2023 2024       # specific seasons

Output (all private - data/ is gitignored):
    data/raw/<name>.tar.xz   downloaded archives (kept so re-runs skip them)
    data/raw/<name>.csv      extracted CSVs
    data/nba.db              SQLite: tables pbp, possessions
"""

import sqlite3
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests

LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"
SEASONS = list(range(2015, 2025))      # 2015-16 .. 2024-25
SOURCES = {"nbastats": "pbp", "pbpstats": "possessions"}
RAW = Path("data/raw")
DB = Path("data/nba.db")


# --- Safety limits. We only ever fetch from this one GitHub repo, and we
# refuse anything unexpectedly large or that isn't a plain CSV. The files are
# treated purely as data: nothing downloaded is ever executed.
ALLOWED_PREFIX = "https://github.com/shufinskiy/nba_data/"
MAX_ARCHIVE_BYTES = 200 * 1024**2      # real archives are ~10-15 MB
MAX_CSV_BYTES = 3 * 1024**3            # real CSVs are a few hundred MB
XZ_MAGIC = b"\xfd7zXZ\x00"             # first 6 bytes of every .xz file


def file_urls():
    """list_data.txt maps names to download links, one 'name=url' per line."""
    r = requests.get(LIST_URL, timeout=30)
    r.raise_for_status()
    urls = {}
    for line in r.text.splitlines():
        if "=" in line:
            name, url = line.strip().split("=", 1)
            urls[name] = url
    return urls


def download(name, url):
    if not url.startswith(ALLOWED_PREFIX):
        raise RuntimeError(f"{name}: unexpected download link {url!r} - not downloading")
    path = RAW / f"{name}.tar.xz"
    if not (path.exists() and path.stat().st_size > 1000):
        print(f"  downloading {name} ...", flush=True)
        tmp = path.with_suffix(".part")     # so a half-finished download is never reused
        size = 0
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise RuntimeError(f"{name}: download is bigger than expected - stopped")
                    f.write(chunk)
        if size < 1000:
            raise RuntimeError(f"{name}: download looks wrong ({size} bytes)")
        tmp.replace(path)
    with open(path, "rb") as f:
        if f.read(6) != XZ_MAGIC:
            raise RuntimeError(f"{path} is not an .xz archive - delete it and retry")
    return path


def extract_csv(archive, name):
    """Extract only the CSV, safely.

    - The archive must contain exactly one thing: a regular .csv file
      (no folders, links, scripts or anything else).
    - We ignore the file name stored in the archive and always write to
      data/raw/<name>.csv, so nothing can land outside data/raw.
    - We stop if it unpacks to more than MAX_CSV_BYTES (a "zip bomb" guard).
    """
    out = RAW / f"{name}.csv"
    if out.exists():
        return out
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getmembers()
        if len(members) != 1 or not members[0].isreg() or not members[0].name.endswith(".csv"):
            listing = [(m.name, "file" if m.isreg() else "NOT a plain file") for m in members[:10]]
            raise RuntimeError(f"{archive}: expected exactly one CSV file, found {listing}")
        if members[0].size > MAX_CSV_BYTES:
            raise RuntimeError(f"{archive}: CSV is bigger than expected - not extracting")
        src = tar.extractfile(members[0])
        tmp = out.with_suffix(".part")
        written = 0
        with open(tmp, "wb") as f:
            while chunk := src.read(1 << 20):
                written += len(chunk)
                if written > MAX_CSV_BYTES:
                    raise RuntimeError(f"{archive}: CSV is bigger than expected - stopped")
                f.write(chunk)
        tmp.replace(out)
    return out


def main():
    seasons = [int(s) for s in sys.argv[1:]] or SEASONS
    RAW.mkdir(parents=True, exist_ok=True)
    urls = file_urls()
    con = sqlite3.connect(DB)

    for source, table in SOURCES.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        for season in seasons:
            for playoffs, tag in ((0, ""), (1, "po_")):          # regular season, then playoffs
                name = f"{source}_{tag}{season}"
                if name not in urls:
                    print(f"  ! {name} not in the dataset list - skipping")
                    continue
                csv = extract_csv(download(name, urls[name]), name)
                rows = 0
                for chunk in pd.read_csv(csv, chunksize=200_000, low_memory=False):
                    chunk.columns = [c.lower() for c in chunk.columns]
                    chunk["season"] = season
                    chunk["playoffs"] = playoffs
                    chunk.to_sql(table, con, if_exists="append", index=False)
                    rows += len(chunk)
                print(f"  {name}: {rows:,} rows -> {table}")
        con.commit()

    for table in SOURCES.values():
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n{table}: {n:,} rows\n  columns: {', '.join(cols)}")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pbp_game ON pbp(game_id)")
    con.close()
    print("\nDone. Next: python build_possessions.py")


if __name__ == "__main__":
    main()
