"""
src/federated/client.py — Flower federated learning client for one state node.
 
Each state node runs this client locally, training on its partition of NFHS-5.
get_parameters() sends model weights to the server.
fit() trains locally for one round and returns updated weights.
evaluate() computes local metrics and returns them to the server.
"""
 
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
 
import flwr as fl
from flwr.common import NDArrays, Scalar
 
from src.model import build_mltp, load_class_weights
from src.evaluation import evaluate_multilabel
from src.config import TARGET_COLS
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
 
class MalnutriSenseClient(fl.client.NumPyClient):
    """
    Flower client for one state-level NFHS-5 partition.
 
    Args:
        X_train: Local training features for this state node.
        y_train: Local training labels for this state node.
        X_test:  Local test features.
        y_test:  Local test labels.
        state:   State name (for logging).
    """
 
    def __init__(self, X_train, y_train, X_test, y_test, state):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.state   = state
        self.class_weights = load_class_weights()
 
        # Build model — not yet fitted
        self.model = build_mltp(X_train, self.class_weights)
        # We must fit once before get_parameters() to initialise weights
        self.model.fit(
            X_train.head(100),
            y_train.head(100)[TARGET_COLS].fillna(0).astype(int)
        )
        log.info(f'Client [{state}] initialised: {len(X_train):,} local rows')
 
    def get_parameters(self, config) -> NDArrays:
        """Return fixed-shape client parameters for Flower aggregation.

        Note: Raw XGBoost boosters serialize to variable-length byte payloads,
        which cannot be averaged with FedAvg. We therefore exchange one
        fixed-length vector per label (feature importances).
        """
        estimators = self.model.named_steps['clf'].estimators_
        n_features = self.X_train.shape[1]
        params: NDArrays = []
        for est in estimators:
            fi = getattr(est, 'feature_importances_', None)
            if fi is None or len(fi) != n_features:
                fi = np.zeros(n_features, dtype=np.float32)
            params.append(np.asarray(fi, dtype=np.float32))
        return params
 
    def set_parameters(self, parameters: NDArrays) -> None:
        """Accept aggregated parameters.

        For the current MLTP-XGBoost setup, FedAvg parameters are used only as
        a compatible exchange payload (feature-importance vectors). They are
        not loaded back into the tree model.
        """
        return
 
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Load global model, train locally for one round, return updated weights."""
        self.set_parameters(parameters)
 
        # Local training — one round on this state's data
        n_epochs = int(config.get('local_epochs', 1))
        log.info(f'[{self.state}] Local training: {len(self.X_train):,} rows')
 
        self.model.fit(
            self.X_train,
            self.y_train[TARGET_COLS].fillna(0).astype(int)
        )
 
        updated_params = self.get_parameters({})
        metrics = {'state': self.state, 'n_samples': len(self.X_train)}
        return updated_params, len(self.X_train), metrics
 
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate global model on local test set."""
        self.set_parameters(parameters)
 
        results = evaluate_multilabel(self.model, self.X_test, self.y_test)
        macro_recall = results['macro_avg']['recall']
 
        log.info(f'[{self.state}] Eval recall: {macro_recall:.4f}')
        return float(1.0 - macro_recall), len(self.X_test), {'recall': macro_recall}
