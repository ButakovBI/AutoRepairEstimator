"""VK API inline-keyboard limits.

Centralising these constants avoids sprinkling magic numbers across keyboard
factories. VK returns ``VKAPIError_911`` when any of these is exceeded, and
historically we have hit both the per-keyboard button cap and the per-row
cap, so we keep the tightest documented limits here.
"""

from __future__ import annotations

from typing import Final

VK_INLINE_MAX_BUTTONS: Final[int] = 10
VK_INLINE_MAX_ROWS: Final[int] = 6
VK_INLINE_DEFAULT_BUTTONS_PER_ROW: Final[int] = 2
