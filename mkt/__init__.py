"""mkt -- shared library for the India + Global market analysis notebooks.

The notebooks stay thin and readable; all fetching, caching, alignment and
metric computation lives here, so a data-source fix happens in one place.

Typical notebook header::

    import sys; sys.path.insert(0, "..")
    from mkt import config, cache, fetch, align, indicators as ind, viz
    viz.setup()

Notebooks 01-06 are the research chain; 07-11 are the fund layer built on top of
it (``backtest``, ``quality``, ``portfolio``, ``risk``, ``derivatives``); 12-14 are
the institutional layer (``perf``, ``costs``, ``validation``, ``factors``,
``margin``, ``limits``, ``manifest``, ``report``).

The institutional layer exists to answer the questions an allocator asks that the
research chain cannot: is the Sharpe measured against cash, does the signal survive
the number of tests that found it, what is the book actually a bet on, can the
account carry it, and can any of it be reproduced next quarter.
"""
from __future__ import annotations

__version__ = "2.0.0"

from . import (align, backtest, cache, config, costs, dashboard, derivatives,
               factors, fetch, fundamentals, indicators, limits, manifest,
               margin, perf, portfolio, quality, report, risk, score,
               validation, viz)

__all__ = [
    "align", "backtest", "cache", "config", "costs", "dashboard", "derivatives",
    "factors", "fetch", "fundamentals", "indicators", "limits", "manifest",
    "margin", "perf", "portfolio", "quality", "report", "risk", "score",
    "validation", "viz",
]


def bootstrap(refresh: bool = False):
    """One-line notebook setup: theme, cache mode, and a stamped header."""
    from datetime import datetime

    cache.set_refresh(refresh)
    viz.setup()
    print(f"mkt v{__version__} | run {datetime.now():%Y-%m-%d %H:%M} | "
          f"REFRESH={cache.REFRESH} | cache={config.CACHE_DIR}")
