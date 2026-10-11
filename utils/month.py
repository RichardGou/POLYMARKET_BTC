"""Dates of the one-month study.

Labels are Gamma windows that start between 11 September and 11 October.
The oracle formula (weights, base, lags) and the probability model are
fitted on the first 20 days. The last 10 days are only used for scoring
and for the trading simulations, together with the Polymarket fills.
"""

from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABEL_START = datetime(2026, 9, 11, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 10, 1, tzinfo=timezone.utc)
TEST_START = TRAIN_END
TEST_END = datetime(2026, 10, 11, tzinfo=timezone.utc)
LABEL_END = TEST_END

FIRST_PRICE_DAY = date(2026, 9, 10)
LAST_PRICE_DAY = date(2026, 10, 10)
