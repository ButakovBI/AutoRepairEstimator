"""Re-export of user-facing labels from the domain layer.

The canonical source is ``backend.domain.value_objects.labels``. This
module stays as a thin re-export so existing bot imports keep working
without a cross-cutting refactor.
"""

from auto_repair_estimator.backend.domain.value_objects.labels import (
    DAMAGE_LABELS,
    PART_LABELS,
)

__all__ = ["DAMAGE_LABELS", "PART_LABELS"]
