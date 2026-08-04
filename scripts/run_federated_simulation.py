"""
scripts/run_federated_simulation.py — Run 10-round Flower federated simulation.
 
Simulates all 5 state nodes locally — no real network required.
Runs 10 federation rounds of FedAvg.
Saves convergence metrics to reports/tables/fl_convergence.csv.
 
Usage: python3 scripts/run_federated_simulation.py
"""
 
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
 
from src.config import PROCESSED_DIR, TABLES_DIR, TARGET_COLS
from src.federated.client import MalnutriSenseClient
from src.federated.partition import create_partitions, FL_NODES
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
NUM_ROUNDS = 10
NUM_CLIENTS = 5


def _normalize_column_name(col):
    return re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower()).strip('_')


def _resolve_column(df, candidates):
    normalized_map = {_normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None
 
def main():
    # Load full cleaned dataset
    df = pd.read_csv(PROCESSED_DIR / 'nfhs5_cleaned.csv')
    log.info(f'Dataset loaded: {len(df):,} rows')

    # Ensure partition key exists with expected name
    state_col = _resolve_column(df, ['V024', 'v024', 'state_code', 'state', 'state_name', 'stateid'])
    if state_col is None:
        raise KeyError(
            f"State column not found. Expected one of: V024, v024, state_code, state, state_name, stateid. "
            f"Available columns: {list(df.columns)}"
        )
    if state_col != 'V024':
        df = df.rename(columns={state_col: 'V024'})
 
    # Create 5 state partitions
    partitions = create_partitions(df)

    # Ensure all clients share identical feature space for parameter aggregation
    if not partitions:
        raise RuntimeError('No valid state partitions were created.')

    global_feature_cols = sorted(
        set().union(*[set(X_tr.columns) for (X_tr, _X_te, _y_tr, _y_te) in partitions.values()])
    )

    aligned_partitions = {}
    for state, (X_tr, X_te, y_tr, y_te) in partitions.items():
        X_tr_aligned = X_tr.reindex(columns=global_feature_cols, fill_value=0)
        X_te_aligned = X_te.reindex(columns=global_feature_cols, fill_value=0)
        aligned_partitions[state] = (X_tr_aligned, X_te_aligned, y_tr, y_te)
    partitions = aligned_partitions
 
    # Record convergence per round
    convergence_log = []
 
    def client_fn(cid: str) -> fl.client.Client:
        '''Create a Flower client for state node cid.'''
        state_names = list(partitions.keys())
        state = state_names[int(cid) % len(state_names)]
        X_tr, X_te, y_tr, y_te = partitions[state]
        return MalnutriSenseClient(X_tr, y_tr, X_te, y_te, state).to_client()
 
    # FedAvg strategy — weighted by number of local samples
    strategy = FedAvg(
        fraction_fit=1.0,         # use all clients each round
        fraction_evaluate=1.0,    # evaluate on all clients
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
 
    # Run simulation
    log.info(f'Starting FL simulation: {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients')
    history = start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
 
    # Save convergence metrics
    rounds_data = []
    losses_by_round = {}
    for rnd, loss in getattr(history, 'losses_distributed', []):
        losses_by_round[int(rnd)] = loss

    fit_metrics_by_round = {}
    for metric_name, metric_series in getattr(history, 'metrics_distributed_fit', {}).items():
        for rnd, value in metric_series:
            fit_metrics_by_round.setdefault(int(rnd), {})[metric_name] = value

    all_rounds = sorted(set(losses_by_round.keys()) | set(fit_metrics_by_round.keys()))
    for rnd in all_rounds:
        rounds_data.append(
            {
                'round': rnd,
                'distributed_loss': losses_by_round.get(rnd),
                'distributed_fit_metrics': str(fit_metrics_by_round.get(rnd, {})),
            }
        )
 
    conv_df = pd.DataFrame(rounds_data)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    conv_df.to_csv(TABLES_DIR / 'fl_convergence.csv', index=False)
 
    print(f'Federated simulation complete: {NUM_ROUNDS} rounds')
    print(f'Convergence log: reports/tables/fl_convergence.csv')
 
if __name__ == '__main__':
    main()
