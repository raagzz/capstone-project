from typing import Optional, Tuple, List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from modAL.models import ActiveLearner
import logging
from tqdm import tqdm
from strategies import ActiveLearningStrategy, get_query_strategy
from visualization import Visualizer
from uncertainty_handler import UncertaintyHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearningTool:
    """A tool for prototyping active learning strategies with scikit-learn models."""

    def __init__(
        self,
        model: BaseEstimator,
        strategy: ActiveLearningStrategy,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_pool: np.ndarray,
        y_pool: Optional[np.ndarray] = None,
        n_queries: int = 50,
        uncertainty_threshold: float = 0.95,
        n_folds: int = 5,
        visualize: bool = True,
        simulate_human: bool = False,
    ):
        self.strategy = strategy
        self.n_queries = n_queries
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy() if y_pool is not None else None
        self.n_folds = n_folds
        self.visualize = visualize
        self.simulate_human = simulate_human
        self.human_query_log = []

        # Initialize components
        self.uncertainty_handler = UncertaintyHandler(strategy, uncertainty_threshold)
        self.visualizer = Visualizer() if visualize else None

        # Initialize the active learner
        self.learner = ActiveLearner(
            estimator=model,
            query_strategy=get_query_strategy(strategy),
            X_training=X_train,
            y_training=y_train,
        )

        # Performance tracking
        self.performance_history = []
        self.query_history = []

    def _get_human_label(
        self, instance: np.ndarray, true_label: Optional[float] = None
    ) -> np.ndarray:
        """Get label from human for highly uncertain instances."""
        if self.simulate_human and true_label is not None:
            # Log the simulated human interaction
            self.human_query_log.append(
                {
                    "instance": instance.tolist(),
                    "true_label": true_label,
                    "uncertainty": self.query_history[-1]
                    if self.query_history
                    else None,
                }
            )
            logger.info(f"Simulated human queried for label. True label: {true_label}")
            return np.array([true_label])
        else:
            print("\nHuman input needed for uncertain instance:")
            print(f"Instance data: {instance}")
            label = input("Please provide the correct label: ")
            return np.array([float(label)])

    def get_human_query_log(self) -> List[dict]:
        """Return the log of all simulated human queries."""
        return self.human_query_log

    def cross_validate(self) -> List[float]:
        """Perform cross-validation on the current model."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in kf.split(self.learner.X_training):
            X_train = self.learner.X_training[train_idx]
            y_train = self.learner.y_training[train_idx]
            X_val = self.learner.X_training[val_idx]
            y_val = self.learner.y_training[val_idx]

            temp_learner = ActiveLearner(
                estimator=self.learner.estimator.__class__(),
                query_strategy=get_query_strategy(self.strategy),
                X_training=X_train,
                y_training=y_train,
            )

            score = temp_learner.score(X_val, y_val)
            cv_scores.append(score)

        return cv_scores

    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Train the model using active learning."""
        pbar = tqdm(total=self.n_queries, desc="Active Learning Progress")

        for idx in range(self.n_queries):
            if len(self.X_pool) == 0:
                logger.info("Pool exhausted. Stopping training.")
                break

            # Query the most informative instance
            query_idx, uncertainty = self.learner.query(self.X_pool)

            # Store uncertainty for visualization
            self.query_history.append(float(np.max(uncertainty)))

            # Get the instance to be labeled
            instance = self.X_pool[query_idx]

            if self.uncertainty_handler.should_query_human(uncertainty):
                label = self._get_human_label(
                    instance,
                    true_label=self.y_pool[query_idx]
                    if self.simulate_human and self.y_pool is not None
                    else None,
                )
            else:
                # Use pool labels if available, otherwise get from human
                if self.y_pool is not None:
                    label = self.y_pool[query_idx]
                else:
                    label = self._get_human_label(instance)

            self.learner.teach(
                X=instance.reshape(1, -1),
                y=np.array([label]).reshape(
                    1,
                ),
            )

            # Remove the queried instance from the pool
            self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
            if self.y_pool is not None:
                self.y_pool = np.delete(self.y_pool, query_idx)

            # Track performance if we have true labels
            if self.y_pool is not None:
                score = self.learner.score(self.X_pool, self.y_pool)
                self.performance_history.append(score)
                logger.info(
                    f"Query {idx + 1}/{self.n_queries}, Performance: {score:.4f}"
                )

            pbar.update(1)

        pbar.close()

        if self.visualize:
            self.visualizer.plot_learning_curve(self.performance_history)
            self.visualizer.plot_query_distribution(self.query_history)
            self.visualizer.plot_cv_results(self.cross_validate())

        return self.X_pool, self.y_pool
