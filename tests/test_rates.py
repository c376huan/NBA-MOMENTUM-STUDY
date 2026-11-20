"""
Check rates.py against the slowest, most obvious way of doing the same thing.

    python tests/test_rates.py        (also runs under pytest)

Three checks:
 1. On a small random league, the fast class gives the same numbers as a
    plain possession-by-possession loop, forwards and backwards, for short
    and long half-lives.
 2. The same on a league as big as the real one. This is the check that the
    earlier formula failed: with tens of thousands of team-games and a short
    half-life it returned 0 for most of every game.
 3. A team's rate does not move on the other team's possessions, and every
    game starts from 0.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rates import Rates  # noqa: E402


def fake_league(games, rng, low=150, high=260):
    """Random games: who has the ball (alternating, with some repeats) and a result for every possession."""
    lengths = rng.integers(low, high, games)
    gidx = np.repeat(np.arange(games), lengths)
    n = len(gidx)
    flips = rng.random(n) < 0.93                      # possession usually changes hands
    home = np.zeros(n, bool)
    state = True
    start = np.r_[True, gidx[1:] != gidx[:-1]]
    for i in range(n):
        state = (rng.random() < 0.5) if start[i] else (not state if flips[i] else state)
        home[i] = state
    x = rng.normal(0, 115, n)                         # like real results: points per 100 vs expectation
    return gidx, home, x


def slow_rates(gidx, home, x, half_life, backwards):
    """(home rate, away rate) after each possession, by the plainest possible loop."""
    keep = 0.5 ** (1 / half_life)
    n = len(x)
    out_home, out_away = np.zeros(n), np.zeros(n)
    idx = range(n - 1, -1, -1) if backwards else range(n)
    r_home = r_away = 0.0
    previous_game = None
    for i in idx:
        if gidx[i] != previous_game:
            r_home = r_away = 0.0
            previous_game = gidx[i]
        if home[i]:
            r_home = keep * r_home + (1 - keep) * x[i]
        else:
            r_away = keep * r_away + (1 - keep) * x[i]
        out_home[i], out_away[i] = r_home, r_away
    return out_home, out_away


def test_matches_plain_loop_small():
    rng = np.random.default_rng(1)
    gidx, home, x = fake_league(40, rng)
    for backwards in (False, True):
        fast = Rates(gidx, home, backwards_in_time=backwards)
        for half_life in (0.5, 1.0, 2.0, 2.8, 4.0, 16.0, 64.0):
            got = fast(x, half_life)
            want = slow_rates(gidx, home, x, half_life, backwards)
            for g, w in zip(got, want):
                assert np.abs(g - w).max() < 1e-9, (half_life, backwards, np.abs(g - w).max())


def test_no_loss_of_precision_on_a_full_size_league():
    rng = np.random.default_rng(2)
    gidx, home, x = fake_league(12_000, rng)          # about as many games as the ten real seasons
    fast = Rates(gidx, home)
    late = gidx >= gidx.max() - 30                    # the last games are where a running total would be largest
    for half_life in (0.5, 1.0, 2.0):
        got = fast(x, half_life)
        want = slow_rates(gidx[late], home[late], x[late], half_life, backwards=False)
        for g, w in zip(got, want):
            assert np.abs(g[late] - w).max() < 1e-9, (half_life, np.abs(g[late] - w).max())
        assert (got[0][late & home] == 0).mean() < 0.01      # the old formula returned exactly 0 for most of these


def test_only_own_possessions_move_a_rate_and_games_start_at_zero():
    rng = np.random.default_rng(3)
    gidx, home, x = fake_league(20, rng)
    r_home, r_away = Rates(gidx, home)(x, 3.0)
    same_game = np.r_[False, gidx[1:] == gidx[:-1]]
    away_ball = ~home & same_game
    assert np.allclose(r_home[away_ball], np.r_[0.0, r_home[:-1]][away_ball])    # unchanged while the other team has it
    first = ~same_game
    keep = 0.5 ** (1 / 3.0)
    assert np.allclose(np.where(home, r_home, r_away)[first], (1 - keep) * x[first])   # first possession: from 0


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
    print("All rate checks passed.")
