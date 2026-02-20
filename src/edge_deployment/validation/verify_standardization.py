#!/usr/bin/env python3
"""
Verifica che la standardizzazione on-board STM32 sia corretta
confrontando con la standardizzazione Python
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Percorsi
CSV_PATH = "/home/cardigun/Scrivania/Progetto SAFE/NOW/data/processed_streaming_row_continuous.csv"
JSON_PATH = "/home/cardigun/Scrivania/Progetto SAFE/NOW/prompt_3/buffermodelli/mt500_b0p6/rep/rep_balance_stats.json"
RAW_PATH = "win_raw.csv"
STD_REF_PATH = "win_std_reference.csv"

print("=" * 60)
print("VERIFICA STANDARDIZZAZIONE ON-BOARD STM32")
print("=" * 60)

# 1. Carica dataset completo
print("\n[1/5] Caricamento dataset completo...")
df_full = pd.read_csv(CSV_PATH)
# Escludi timestamp e label
feature_cols = [col for col in df_full.columns if col not in ['label', 'timestamp']]
X_full = df_full[feature_cols].values
print(f"   ✅ Dataset: {X_full.shape[0]} righe × {X_full.shape[1]} features")

# 2. Fit StandardScaler
print("\n[2/5] Fit StandardScaler su dataset completo...")
scaler = StandardScaler()
scaler.fit(X_full)
print(f"   ✅ Scaler fitted")

# 3. Carica win_raw.csv e standardizza
print("\n[3/5] Standardizzazione di win_raw.csv...")
win_raw = pd.read_csv(RAW_PATH, header=None).values
win_std_computed = scaler.transform(win_raw)
print(f"   ✅ Standardizzato: {win_std_computed.shape}")
print(f"   Range RAW: [{win_raw.min():.3f}, {win_raw.max():.3f}]")
print(f"   Range STD: [{win_std_computed.min():.3f}, {win_std_computed.max():.3f}]")

# 4. Confronta con win_std_reference.csv
print("\n[4/5] Confronto con win_std_reference.csv...")
win_std_ref = pd.read_csv(STD_REF_PATH, header=None).values
diff = np.abs(win_std_computed - win_std_ref)
max_diff = diff.max()
mean_diff = diff.mean()
print(f"   Max differenza: {max_diff:.6e}")
print(f"   Mean differenza: {mean_diff:.6e}")

if max_diff < 1e-5:
    print("   ✅✅✅ IDENTICI! (diff < 1e-5)")
else:
    print(f"   ⚠️  DIFFERENZE RILEVATE!")
    print(f"   Posizione max diff: {np.unravel_index(diff.argmax(), diff.shape)}")

# 5. Verifica parametri da JSON
print("\n[5/5] Verifica parametri da JSON...")
with open(JSON_PATH, 'r') as f:
    stats = json.load(f)

mean_json = np.array(stats['scaler']['mean'])
std_json = np.array(stats['scaler']['scale'])  # 'scale' non 'std'!

diff_mean = np.abs(scaler.mean_ - mean_json)
diff_std = np.abs(scaler.scale_ - std_json)

print(f"   Max diff MEAN (Scaler vs JSON): {diff_mean.max():.6e}")
print(f"   Max diff STD  (Scaler vs JSON): {diff_std.max():.6e}")

if diff_mean.max() < 1.5e-5 and diff_std.max() < 1.5e-5:
    print("   ✅✅✅ PARAMETRI JSON CORRETTI!")
else:
    print("   ⚠️  Parametri JSON divergono!")

# 6. Stampa esempio standardizzazione manuale
print("\n" + "=" * 60)
print("ESEMPIO STANDARDIZZAZIONE MANUALE:")
print("=" * 60)
idx = 0  # Prima riga
feat = 0  # Prima feature
raw_val = win_raw[idx, feat]
std_val_computed = win_std_computed[idx, feat]
std_val_ref = win_std_ref[idx, feat]
mean = scaler.mean_[feat]
std = scaler.scale_[feat]

print(f"Riga {idx}, Feature {feat}:")
print(f"  RAW value:        {raw_val:.6f}")
print(f"  MEAN:             {mean:.6f}")
print(f"  STD:              {std:.6f}")
print(f"  Formula:          ({raw_val:.6f} - {mean:.6f}) / {std:.6f}")
print(f"  Computed:         {std_val_computed:.6f}")
print(f"  Reference:        {std_val_ref:.6f}")
print(f"  Difference:       {abs(std_val_computed - std_val_ref):.6e}")

print("\n" + "=" * 60)
print("✅ VERIFICA COMPLETATA")
print("=" * 60)
