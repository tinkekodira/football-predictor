"""Statistical models: Phase 2.

    from fbedge.models import load_training_set, fit_goals_model, fit_count_model

The goals model produces two scoring rates and a full scoreline matrix, from
which every goal-based market is derived. The count models handle corners and
cards, which are overdispersed and therefore need a negative binomial rather
than a Poisson.
"""

from .base import (  # noqa: F401
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_RIDGE,
    InsufficientData,
    TrainingSet,
    load_training_set,
    time_weights,
)
from .counts import CountModel, fit_count_model  # noqa: F401
from .goals import GoalsModel, fit_goals_model, score_matrix_from_rates  # noqa: F401

__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_RIDGE",
    "CountModel",
    "GoalsModel",
    "InsufficientData",
    "TrainingSet",
    "fit_count_model",
    "fit_goals_model",
    "load_training_set",
    "score_matrix_from_rates",
    "time_weights",
]
