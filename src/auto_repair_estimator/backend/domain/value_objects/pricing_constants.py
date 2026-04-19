"""Pricing notes that don't fit the ``pricing_rules`` table.

Two soft-notes drive user-facing messages out of band:

* ``POLISH_*`` — for every detected ``scratch`` the smithy-level estimate
  already assumes painting (the conservative upper-bound price). If the
  scratch is shallow and polishing is enough, we communicate the cheaper
  alternative as an extra line at the end of the estimate. Values here
  match thesis table 5 / 6 exactly: 1 hour, 1 000 RUB per scratch.

* ``TYRE_SHOP_NOTE`` — wheel/tyre problems are explicitly out of the body
  shop's scope per the requirements spec; we route the user to a tyre
  shop instead of producing a priced row.
"""

from __future__ import annotations

from typing import Final

POLISH_HOURS: Final[float] = 1.0
POLISH_COST_RUB: Final[float] = 1000.0

TYRE_SHOP_NOTE: Final[str] = (
    "Для повреждений шин и колёсных дисков кузовной ремонт не применим — "
    "обратитесь в шиномонтаж."
)
