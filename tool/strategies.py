from enum import Enum
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling


class ActiveLearningStrategy(Enum):
    """Enum for different active learning strategies."""

    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    ENTROPY = "entropy"


def get_query_strategy(strategy: ActiveLearningStrategy):
    """Return the query strategy based on the selected enum."""
    strategies = {
        ActiveLearningStrategy.UNCERTAINTY: uncertainty_sampling,
        ActiveLearningStrategy.MARGIN: margin_sampling,
        ActiveLearningStrategy.ENTROPY: entropy_sampling,
    }
    return strategies[strategy]
