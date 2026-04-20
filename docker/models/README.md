# ML Model Weights

The `ml_worker` service mounts this directory at `/app/models:ro` (see
`docker/docker-compose.yml`). On startup it loads two YOLOv8-seg networks
from fixed paths:

| File | Purpose | Class names (must match exactly) |
|------|---------|-----------------------------------|
| `damages.pt` | Detects damage instances on a cropped part | `scratch`, `dent`, `paint_chip`, `rust`, `crack`, `broken_glass`, `flat_tire`, `broken_headlight` |
| `parts.pt`   | Segments car parts on the full image | `door`, `front_fender`, `rear_fender`, `trunk`, `hood`, `roof`, `headlight`, `front_windshield`, `rear_windshield`, `side_window`, `wheel`, `bumper` |

Class names are the single source of truth that ties the model to the
`DamageType` / `PartType` enums in
`src/auto_repair_estimator/backend/domain/value_objects/request_enums.py`.
On load the detectors cross-check `model.names` against the enums and log
a warning for any unknown class; unknown classes are also silently skipped
at inference time so the backend never receives an enum value it cannot
store.

## How to deploy new weights

1. Train the model — see `ml/train_damages.py` and `ml/train_parts.py`
   (datasets declared in `ml/data/*.yaml`).
2. Copy the resulting `best.pt` into this directory under the exact name
   the worker expects (`damages.pt` or `parts.pt`).
3. Restart the worker: `docker compose -f docker/docker-compose.yml restart ml_worker`.

Weights are **not** committed to git (see `.gitignore` — `docker/models/*.pt`).
Distribute them out-of-band (artefact registry, S3, etc.) alongside a note
of the git SHA and the commit of the enum definitions they were aligned
with.
