import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from datetime import datetime
from typing import List, Dict, Union, Tuple

class QuerySelector:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.features_info = self._analyze_features()

    def _analyze_features(self) -> Dict:
        features_info = {}
        for column in self.X.columns:
            column_type = self.X[column].dtype
            if np.issubdtype(column_type, np.number):
                features_info[column] = {
                    'type': 'numeric',
                    'min': float(self.X[column].min()),
                    'max': float(self.X[column].max()),
                    'mean': float(self.X[column].mean()),
                    'std': float(self.X[column].std())
                }
            else:
                features_info[column] = {
                    'type': 'categorical',
                    'unique_values': sorted(self.X[column].unique().tolist())
                }
        return features_info

    def get_features_info(self) -> Dict:
        return self.features_info

    def random_selection(self, n_samples: int) -> np.ndarray:
        return np.random.choice(
            len(self.X), size=n_samples, replace=False
        )

    def feature_based_selection(self, query_conditions: List[Dict]) -> np.ndarray:
        mask = np.ones(len(self.X), dtype=bool)
        for condition in query_conditions:
            feature = condition['feature']
            operator = condition['operator']
            value = condition['value']

            if feature not in self.X.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset")

            if operator == 'gt':
                mask &= (self.X[feature] > value)
            elif operator == 'lt':
                mask &= (self.X[feature] < value)
            elif operator == 'eq':
                mask &= (self.X[feature] == value)
            elif operator == 'between':
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError("'between' operator requires a list/tuple of [min, max] values")
                mask &= (self.X[feature] >= value[0]) & (self.X[feature] <= value[1])
            else:
                raise ValueError(f"Unknown operator: {operator}")

        return np.where(mask)[0]

class SISA:
    AVAILABLE_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'logistic_regression': LogisticRegression
    }

    def __init__(self, model_name='random_forest', n_shards=5, n_estimators=100, **model_params):
        self.model_name = model_name
        self.n_shards = n_shards
        self.n_estimators = n_estimators
        self.model_params = model_params
        self.models = []
        self.shard_indices = []
        self.query_selector = None

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")

        if model_name in ['random_forest', 'gradient_boosting']:
            self.trees_per_shard = n_estimators // n_shards
            self.model_params['n_estimators'] = self.trees_per_shard

    def _create_model(self):
        model_class = self.AVAILABLE_MODELS[self.model_name]
        return model_class(**self.model_params)

    def fit(self, X, y):
        self.query_selector = QuerySelector(X, y)
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        shard_size = max(n_samples // self.n_shards, 1)
        self.shard_indices = [
            indices[i:i + shard_size]
            for i in range(0, n_samples, shard_size)
        ]

        self.models = []
        for shard_idx in self.shard_indices:
            if len(shard_idx) > 0:  # Ensure shard is not empty
                model = self._create_model()
                model.fit(X.iloc[shard_idx], y.iloc[shard_idx])
                self.models.append(model)

    def unlearn(self, X, y, selection_strategy='random', strategy_params=None):
        if strategy_params is None:
            strategy_params = {}

        if selection_strategy == 'random':
            n_samples = strategy_params.get('n_samples', 50)
            forgot_indices = self.query_selector.random_selection(n_samples)
        elif selection_strategy == 'feature_based':
            conditions = strategy_params.get('conditions', [])
            forgot_indices = self.query_selector.feature_based_selection(conditions)
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        affected_shards = []
        for shard_id, shard_idx in enumerate(self.shard_indices):
            if any(idx in shard_idx for idx in forgot_indices):
                affected_shards.append(shard_id)

                keep_mask = ~np.isin(shard_idx, forgot_indices)
                new_shard_idx = shard_idx[keep_mask]
                self.shard_indices[shard_id] = new_shard_idx

                if len(new_shard_idx) > 0:  # Ensure non-empty shard
                    model = self._create_model()
                    model.fit(X.iloc[new_shard_idx], y.iloc[new_shard_idx])
                    self.models[shard_id] = model

        return affected_shards, forgot_indices

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return np.array([
            np.bincount(pred.astype(int)).argmax()
            for pred in predictions
        ])

    def get_features_info(self):
        if self.query_selector is None:
            raise ValueError("Fit Model Before Getting Features Info")
        return self.query_selector.get_features_info()

class Analyzer:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, y_true, y_pred, stage):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'classification_report': classification_report(y_true, y_pred)
        }
        self.results[stage] = metrics

    def generate_report(self):
        before = self.results['before_unlearning']
        after = self.results['after_unlearning']

        changes = {
            metric: after[metric] - before[metric]
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        }

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics_before_unlearning': {k: v for k, v in before.items() if k != 'classification_report'},
            'metrics_after_unlearning': {k: v for k, v in after.items() if k != 'classification_report'},
            'metric_changes': changes
        }

        return report

def run_experiment(X_train, X_test, y_train, y_test, model_name='random_forest', selection_strategy='random', strategy_params=None, n_shards=5, n_estimators=100, **model_params):
    sisa = SISA(model_name=model_name, n_shards=n_shards, n_estimators=n_estimators, **model_params)
    analyzer = Analyzer()

    sisa.fit(X_train, y_train)
    initial_pred = sisa.predict(X_test)
    analyzer.evaluate_model(y_test, initial_pred, 'before_unlearning')

    affected_shards, forgot_indices = sisa.unlearn(X_train, y_train, selection_strategy=selection_strategy, strategy_params=strategy_params)
    final_pred = sisa.predict(X_test)
    analyzer.evaluate_model(y_test, final_pred, 'after_unlearning')

    report = analyzer.generate_report()
    report['experiment_details'] = {
        'model_name': model_name,
        'n_shards': n_shards,
        'n_estimators': n_estimators,
        'samples_forgotten': len(forgot_indices),
        'affected_shards': len(affected_shards)
    }

    return report