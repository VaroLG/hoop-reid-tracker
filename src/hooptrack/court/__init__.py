"""Chapter 2 · Court geometry (independent of the tracker).

Everything about the court lives here: keypoint templates per league, the
court-keypoint detector, image <-> court homographies and, later, the top-down
projection of players. Experiments: ``experiments/court/`` (C01, C02, ...);
scripts: ``scripts/court/``; tests: ``tests/court/``.

Dependency rule (enforced by ``tests/court/test_separation.py``):

* ``hooptrack.court`` never imports the tracker (``tracking``, ``reid``,
  ``pipeline``, ...): it can be used, tested and trained on its own.
* The tracker never imports ``hooptrack.court`` directly. When the court
  model is ready, ONE adapter in ``hooptrack/integrations/`` will expose it to
  the pipeline as a filter, behind a config switch, like every other change.
"""

from .template import FT_TO_M, NBA_KEYPOINTS, NON_FLOOR_KEYPOINTS, CourtTemplate, nba_template

__all__ = ["FT_TO_M", "NBA_KEYPOINTS", "NON_FLOOR_KEYPOINTS", "CourtTemplate", "nba_template"]
