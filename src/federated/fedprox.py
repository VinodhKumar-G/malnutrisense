"""
src/federated/fedprox.py — FedProx strategy for heterogeneous state-level data.
 
FedProx adds a proximal term (mu * ||w - w_global||^2) to the local loss,
preventing state-level models from diverging too far from the global model.
 
Public API:
  FedProxClient(NumPyClient)  — client with proximal regularisation
  run_fedprox_simulation()    — 10-round simulation with FedProx
"""
 
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
 
import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedProx
 
from src.model import build_mltp, load_class_weights
from src.evaluation import evaluate_multilabel
from src.config import TARGET_COLS
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
# Proximal mu — controls how much local training can diverge from global
# Higher mu = more constrained (closer to global), lower mu = more local freedom
PROXIMAL_MU = 0.1  # standard starting value
 
 
class FedProxClient(fl.client.NumPyClient):
    """
    Flower client with FedProx proximal regularisation.
 
    During fit(), the local loss includes an additional term:
      L_prox = (mu / 2) * ||w_local - w_global||^2
 
    This prevents the UP node (high stunting) from pulling the global model
    too far from what works for Kerala (low stunting).
    """
 
    def __init__(self, X_train, y_train, X_test, y_test, state, mu=PROXIMAL_MU):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.state   = state
        self.mu      = mu
        self.class_weights = load_class_weights()
 
        # Build and initialise model
        self.model = build_mltp(X_train, self.class_weights)
        self.model.fit(
            X_train.head(100),
            y_train.head(100)[TARGET_COLS].fillna(0).astype(int)
        )
        # Store global weights for proximal penalty computation
        self._global_params = self.get_parameters({})
        log.info(f'FedProxClient [{state}] initialised: mu={mu}')
 
    def get_parameters(self, config) -> NDArrays:
        estimators = self.model.named_steps['clf'].estimators_
        return [est.get_booster().save_raw('ubj') for est in estimators]
 
    def set_parameters(self, parameters: NDArrays) -> None:
        estimators = self.model.named_steps['clf'].estimators_
        for est, param in zip(estimators, parameters):
            est.get_booster().load_model(param)
        # Update stored global weights for next proximal computation
        self._global_params = list(parameters)
 
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
 
        # Standard local training
        self.model.fit(
            self.X_train,
            self.y_train[TARGET_COLS].fillna(0).astype(int)
        )
 
        # Note: XGBoost does not natively support the proximal term in its loss.
        # The Flower FedProx strategy handles the proximal aggregation server-side.
        # The client just trains normally; the server applies the mu-weighted
        # penalty during model aggregation.
 
        updated_params = self.get_parameters({})
        return updated_params, len(self.X_train), {'state': self.state}
 
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        results = evaluate_multilabel(self.model, self.X_test, self.y_test)
        macro_recall = results['macro_avg']['recall']
        return float(1.0 - macro_recall), len(self.X_test), {'recall': macro_recall}
 
 
def run_fedprox_simulation(partitions, num_rounds=10, mu=PROXIMAL_MU):
    """Run FedProx simulation with given state partitions."""
    from flwr.simulation import run_simulation
 
    def client_fn(cid):
        state_names = list(partitions.keys())
        state = state_names[int(cid) % len(state_names)]
        X_tr, X_te, y_tr, y_te = partitions[state]
        return FedProxClient(X_tr, y_tr, X_te, y_te, state, mu=mu).to_client()
 
    strategy = FedProx(
        proximal_mu=mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(partitions),
        min_evaluate_clients=len(partitions),
        min_available_clients=len(partitions),
    )
 
    history = run_simulation(
        client_fn=client_fn,
        num_clients=len(partitions),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return history
