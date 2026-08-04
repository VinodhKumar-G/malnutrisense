# scripts/generate_ablation_table.py
import pandas as pd
from src.config import TABLES_DIR
 
# Load all benchmark results
base     = pd.read_csv(TABLES_DIR/'full_benchmark.csv')
ablation = pd.read_csv(TABLES_DIR/'ablation_results.csv')
 
# Build combined ablation summary
summary = pd.DataFrame({
    'Model version':   ['MLTP (baseline)', 'MLTP + satellite', 'MLTP + velocity', 'MLTP + both'],
    'Features added':  ['None', 'NDVI + SPI', 'delta_HAZ/WAZ/WHZ', 'All satellite + velocity'],
    'Macro Recall':    ['[from CSV]', '[from CSV]', '[from CSV]', '[from CSV]'],
    'Stunted Recall':  ['[from CSV]', '[from CSV]', '[from CSV]', '[from CSV]'],
    'Wasted Recall':   ['[from CSV]', '[from CSV]', '[from CSV]', '[from CSV]'],
})
summary.to_csv(TABLES_DIR/'ablation_full_table.csv', index=False)
print(summary.to_string(index=False))
