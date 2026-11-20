"""
NBA Momentum Study - the recency-weighted scoring rate, in one place.

A team's rate is updated after each of its own offensive possessions:

    rate <- keep * rate + (1 - keep) * (this possession's result)

starting from 0 every game, with keep = 0.5 ** (1 / half_life). Used by
momentum.py (forwards in time = momentum before a possession; backwards in
time = momentum ahead of it).

Why this is a short loop and not one formula. The update can be "unrolled"
into a weighted sum, rate_t = (1 - keep) * sum_j keep^(t-j) * x_j, and that
sum can be computed for every game at once from a single running total of
x_j * keep^(-j). An earlier version did exactly that. It is fast, but it is
not safe: keep^(-j) grows without limit (2^100 after a team's 100th
possession at half-life 1), so the running total over thousands of games
becomes astronomically large, and a computer keeps only about 16 significant
digits. The small numbers at the start of the next game were simply rounded
away. With half-lives under about 2.8 a large share of the rates came out
wrong (41% at half-life 2, 89% at half-life 1), most of them as exactly 0.

The loop below does the update directly. It steps through "a team's k-th
possession of the game" (at most about 140 steps) and updates every
team-game at once in each step, so it is exact and still takes a fraction of
a second for 2.5 million possessions. tests/test_rates.py checks it against
a plain possession-by-possession loop.
"""

import numpy as np


class Rates:
    """Recency-weighted scoring rate of each team, forwards or backwards in time.

    gidx: game number of every possession (0, 0, ..., 1, 1, ...), in game order.
    home: True where the home team has the ball.
    """

    def __init__(self, gidx, home, backwards_in_time=False):
        self.rev = backwards_in_time
        self.n = n = len(gidx)
        pos = np.arange(n)
        team_game = gidx * 2 + home
        tg, self.home, self.g = (a[::-1] if self.rev else a for a in (team_game, home, gidx))
        self.order = np.argsort(tg, kind="stable")                 # each team's own possessions, in order
        tgs = tg[self.order]
        start = np.r_[True, tgs[1:] != tgs[:-1]]
        self.row = np.cumsum(start) - 1                             # which team-game (0, 1, 2 ...)
        self.t = pos - np.maximum.accumulate(np.where(start, pos, 0))   # 0, 1, 2 ... within the team's game
        self.shape = (self.row[-1] + 1, self.t.max() + 1)           # (team-games, most possessions in one)
        self.last_home = np.maximum.accumulate(np.where(self.home, pos, -1))
        self.last_away = np.maximum.accumulate(np.where(~self.home, pos, -1))

    def __call__(self, x, half_life):
        """Returns (home team's rate, away team's rate) after each possession, counting it."""
        keep = 0.5 ** (1 / half_life)
        table = np.zeros(self.shape)                                # one row per team-game, one column per possession
        table[self.row, self.t] = (x[::-1] if self.rev else x)[self.order]
        rate = np.zeros(self.shape[0])
        for k in range(self.shape[1]):                              # every team's k-th possession, all games at once
            rate = keep * rate + (1 - keep) * table[:, k]
            table[:, k] = rate
        own = np.empty(self.n)                                      # rate of the team with the ball, incl. this possession
        own[self.order] = table[self.row, self.t]
        out = []
        for last_idx in (self.last_home, self.last_away):
            ok = (last_idx >= 0) & (self.g[np.maximum(last_idx, 0)] == self.g)
            v = np.where(ok, own[np.maximum(last_idx, 0)], 0.0)
            out.append(v[::-1] if self.rev else v)
        return out
