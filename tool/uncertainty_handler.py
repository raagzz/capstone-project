import numpy as np
from strategies import ActiveLearningStrategy


class UncertaintyHandler:
    def __init__(self, strategy: ActiveLearningStrategy, base_threshold: float = 0.95):
        self.strategy = strategy
        self.base_threshold = base_threshold

    def should_query_human(self, uncertainty: np.ndarray) -> bool:
        """
        Determine if human input should be requested based on uncertainty.
        Uses different thresholds for different strategies.
        """
        if isinstance(uncertainty, np.ndarray):
            if self.strategy == ActiveLearningStrategy.UNCERTAINTY:
                # Query when very high uncertainty
                return uncertainty.max() > self.base_threshold
            elif self.strategy == ActiveLearningStrategy.MARGIN:
                # Query when margin between top classes is very small
                return uncertainty.min() < (1 - self.base_threshold)
            else:  # ENTROPY
                # Query when entropy is very high
                return uncertainty.mean() > self.base_threshold
        return uncertainty > self.base_threshold
