import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
import matplotlib

matplotlib.use("Agg")


class Visualizer:
    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_learning_curve(self, performance_history: List[float]):
        """Plot and save the learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(performance_history) + 1),
            performance_history,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
        )
        plt.grid(True, alpha=0.3)
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        plt.title("Active Learning Performance Over Time")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "learning_curve.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_query_distribution(self, query_history: List[float]):
        """Plot and save the query uncertainty distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=query_history, bins=20, kde=True)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Uncertainty Score")
        plt.ylabel("Count")
        plt.title("Distribution of Query Uncertainties")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "query_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_cv_results(self, cv_scores: List[float]):
        """Plot and save cross-validation results."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=cv_scores)
        plt.grid(True, alpha=0.3)
        plt.ylabel("Score")
        plt.title("Cross-validation Performance Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "cv_results.png", dpi=300, bbox_inches="tight")
        plt.close()
