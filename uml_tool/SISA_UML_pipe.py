##Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import json
from datetime import datetime
from typing import List, Dict, Union, Tuple

##Writing a Class for the Query Selector (different ways of cand unlearning)
class QuerySelector:
    def __init__(self,X:pd.DataFrame,y:pd.Series):
        self.X = X
        self.y = y
        self.features_info = self._analyze_features()

    def _analyze_features(self) -> Dict:
        features_info = {}
        for column in self.X.columns:
            column_type = self.X[column].dtype

            if np.issubdtype(column_type,np.number):
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
    
    def random_selection(self,n_samples:int) -> np.ndarray:
        return np.random.choice(
            len(self.X),size=n_samples,replace=False
        )
    
    def feature_based_selection(self,query_conditions: List[Dict]) -> np.ndarray:
        mask = np.ones(len(self.X),dtype=bool)
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

##Class for the SISA along with qr strat
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
        
        shard_size = n_samples // self.n_shards
        self.shard_indices = [
            indices[i:i+shard_size]
            for i in range(0, n_samples, shard_size)
        ]
        
        self.models = []
        for shard_idx in self.shard_indices:
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
                
                model = self._create_model()
                model.fit(X.iloc[new_shard_idx], y.iloc[new_shard_idx])
                self.models[shard_id] = model
                
        return affected_shards, forgot_indices

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:,i] = model.predict(X)
            
        return np.array([
            np.bincount(pred.astype(int)).argmax()
            for pred in predictions
        ])
    
    def get_features_info(self):
        if self.query_selector is None:
            raise ValueError("Fit Model Before Getting Features Info")
        return self.query_selector.get_features_info()
    
##Class for the Quality Analysis
class Analyzer:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self,y_true,y_pred,stage):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'classification_report': classification_report(y_true, y_pred)
        }
        self.results[stage] = metrics

    def _analyze_impact(self, changes):
        avg_change = np.mean(list(changes.values()))
        if abs(avg_change) < 0.01:
            return "Minimal impact: Model performance remained stable after unlearning"
        elif avg_change > 0:
            return "Positive impact: Model performance improved slightly after unlearning"
        else:
            return "Negative impact: Model performance decreased slightly after unlearning"
    
    def _assess_stability(self, changes):
        max_change = max(abs(v) for v in changes.values())
        if max_change < 0.03:
            return "High stability: All metrics remained within 3% of original values"
        elif max_change < 0.05:
            return "Moderate stability: Some metrics changed by up to 5%"
        else:
            return "Low stability: Significant changes observed in some metrics"
    
    def generate_report(self):
        before = self.results['before_unlearning']
        after = self.results['after_unlearning']

        changes = {
            metric: after[metric] - before[metric]
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        }

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics_before_unlearning': {
                k: v for k, v in before.items() if k != 'classification_report'
            },
            'metrics_after_unlearning': {
                k: v for k, v in after.items() if k != 'classification_report'
            },
            'metric_changes': changes,
            'analysis_summary': {
                'performance_impact': self._analyze_impact(changes),
                'stability_assessment': self._assess_stability(changes)
            }
        }

        return report
    
##Creating a Dummy Function for the data generation
def create_sample():
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            'Income': np.random.normal(50000,2000,n_samples),
            'Age': np.random.normal(40,15,n_samples),
            'Work_yrs': np.random.normal(15,3,n_samples),
            'debt_ratio': np.random.normal(0.3,0.1,n_samples),
            'credit_score': np.random.normal(700,100,n_samples)
        }
    )

    data['Target'] = (
        (data['Income'] > 45500) &
        (data['credit_score'] > 650) &
        (data['debt_ratio'] < 0.4)
    ).astype(int)

    X = data.drop('Target',axis=1)
    y = data['Target']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    return X_train,X_test,y_train,y_test

##Run the Tool Kit Function
def run_experiment(X_train, X_test, y_train, y_test, model_name='random_forest', 
                  selection_strategy='random', strategy_params=None, n_shards=5, 
                  n_estimators=100, **model_params):
    sisa = SISA(
        model_name=model_name,
        n_shards=n_shards,
        n_estimators=n_estimators,
        **model_params
    )

    analyzer = Analyzer()
    sisa.fit(X_train, y_train)
    initial_pred = sisa.predict(X_test)

    analyzer.evaluate_model(y_test, initial_pred, 'before_unlearning')

    affected_shards, forgot_indices = sisa.unlearn(
        X_train, y_train,
        selection_strategy=selection_strategy,
        strategy_params=strategy_params
    )

    final_pred = sisa.predict(X_test)
    analyzer.evaluate_model(y_test, final_pred, 'after_unlearning')

    ##So -> Generate Report
    report = analyzer.generate_report()
    report['experiment_details'] = {
        'model_name': model_name,
        'n_shards': n_shards,
        'n_estimators': n_estimators,
        'samples_forgotten': len(forgot_indices),
        'affected_shards': len(affected_shards)
    }

    return report

X_train, X_test, y_train, y_test = create_sample()

sisa = SISA(model_name='random_forest', n_shards=5, n_estimators=100)
sisa.fit(X_train, y_train)

features_info = sisa.get_features_info()
print("Available features for querying:", json.dumps(features_info, indent=2))

random_report = run_experiment(
    X_train, X_test, y_train, y_test,
    selection_strategy='random',
    strategy_params={'n_samples': 50}
)

feature_based_report = run_experiment(
    X_train, X_test, y_train, y_test,
    selection_strategy='feature_based',
    strategy_params={
        'conditions': [
            {
                'feature': 'Income',
                'operator': 'between',
                'value': [45000, 55000]
            },
            {
                'feature': 'credit_score',
                'operator': 'gt',
                'value': 700
            }
        ]
    }
)

print("\nRandom Selection Report:")
print()
print(json.dumps(random_report, indent=2))

print("\nFeature-based Selection Report:")
print()
print(json.dumps(feature_based_report, indent=2))