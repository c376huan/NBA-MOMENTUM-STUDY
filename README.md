# Is momentum real in the NBA?

A study to see if momentum is real in basketball. If it is: who are the
biggest momentum creators, and who are the momentum destroyers?

Uses play-by-play data from every NBA regular-season and playoff game, 2015-16 to 2024-25.

**Status:** work in progress.

## Reproduce
Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`). Then:

    python run_all.py              # downloads the data (about 3 GB) and runs every step
    python run_all.py --fake 600   # a small simulated league instead: checks every step runs
    python tests/test_rates.py     # quick check of the momentum calculation

The steps and their order are listed at the top of `run_all.py`; each script explains
what it does at the top of the file.

## Data
Play-by-play and possession data from the
[shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) dataset
(Apache-2.0), compiled from stats.nba.com and pbpstats.com. Raw data is not
included in this repo. Head coaches by season are from
[Basketball-Reference.com](https://www.basketball-reference.com/).
Independent analysis; not affiliated with the NBA or any team.
