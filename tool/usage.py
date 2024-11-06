if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from active_learner import ActiveLearningStrategy, ActiveLearningTool

    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target

    initial_indices = []
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        initial_indices.extend(np.random.choice(class_indices, size=2, replace=False))

    initial_indices = np.array(initial_indices)
    X_train = X[initial_indices]
    y_train = y[initial_indices]
    X_pool = np.delete(X, initial_indices, axis=0)
    y_pool = np.delete(y, initial_indices)

    al_tool = ActiveLearningTool(
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        strategy=ActiveLearningStrategy.UNCERTAINTY,
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        n_queries=20,
        uncertainty_threshold=0.8,  # Lower threshold to trigger more human queries
        n_folds=5,
        visualize=True,
        simulate_human=True,  # Enable human simulation
    )

    remaining_X_pool, remaining_y_pool = al_tool.train()

    print("\nTraining Results:")
    print("-" * 50)
    performance = al_tool.performance_history
    print("Performance history:")
    for i, score in enumerate(performance, 1):
        print(f"Query {i}: {score:.4f}")

    cv_scores = al_tool.cross_validate()
    print("\nCross-validation Results:")
    print("-" * 50)
    print(f"Individual scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    human_queries = al_tool.get_human_query_log()
    print("\nHuman Query Summary:")
    print("-" * 50)
    print(f"Total number of human queries: {len(human_queries)}")
    print("\nDetailed Query Information:")
    for i, query in enumerate(human_queries, 1):
        print(f"\nQuery {i}:")
        print(f"  Uncertainty: {query['uncertainty']:.4f}")

        true_label_idx = int(
            query["true_label"].item()
            if isinstance(query["true_label"], np.ndarray)
            else query["true_label"]
        )
        print(f"  True label: {iris.target_names[true_label_idx]}")

        features = np.array(query["instance"])
        feature_names = iris.feature_names
        print("  Features:")
        for fname, fvalue in zip(feature_names, features):
            print(f"    {fname}: {fvalue:}")
