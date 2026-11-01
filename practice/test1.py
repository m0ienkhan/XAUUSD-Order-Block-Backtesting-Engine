import dukascopy_python as dp
from datetime import datetime

data = dp.fetch(
    "XAUUSD",
    dp.INTERVAL_MIN_15,
    dp.OFFER_SIDE_BID,
    datetime(2025, 1, 1),
    datetime(2025, 1, 2)
)

print(data)