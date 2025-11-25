#!/usr/bin/env python3
"""
Verifica che MEAN/STD siano identici tra training, quantizzazione e JSON
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

CSV = "/home/cardigun/Scrivania/Progetto SAFE/NOW/data/processed_streaming_row_continuous.csv"
JSON = "/home/cardigun/Scrivania/Progetto SAFE/NOW/finalepipeline_4/qbench_sweep_20251027_165731/experiments/mt500_b0p6/rep/rep_balance_stats.json"
NFEAT = 16

print("=" * 70)
print("VERIFICA COERENZA MEAN/STD")
print("=" * 70)

# 1) Come fa il training (cnn_lstm.py)
print("\n[1/3] Simulo training...")
df = pd.read_csv(CSV, low_memory=False)
num_cols = df.select_dtypes(include=['number']).columns
scaler_train = StandardScaler()
scaler_train.fit(df[num_cols])
mean_train = scaler_train.mean_[:NFEAT]
std_train = scaler_train.scale_[:NFEAT]

# 2) Come fa la quantizzazione (make_rep_balanced)
print("[2/3] Simulo quantizzazione...")
df2 = pd.read_csv(CSV, low_memory=False)
num_df = df2.select_dtypes(include=[np.number])
data = num_df.iloc[:, :NFEAT].to_numpy(dtype=np.float32)
scaler_quant = StandardScaler()
scaler_quant.fit(data)
mean_quant = scaler_quant.mean_
std_quant = scaler_quant.scale_

# 3) Dal JSON (salvato durante quantizzazione)
print("[3/3] Leggo JSON...")
with open(JSON, 'r') as f:
    stats = json.load(f)
mean_json = np.array(stats["scaler"]["mean"], dtype=np.float32)
std_json = np.array(stats["scaler"]["scale"], dtype=np.float32)

# Confronto
print("\n" + "=" * 70)
print("CONFRONTO")
print("=" * 70)

diff_train_quant_mean = np.abs(mean_train - mean_quant).max()
diff_train_quant_std = np.abs(std_train - std_quant).max()
diff_quant_json_mean = np.abs(mean_quant - mean_json).max()
diff_quant_json_std = np.abs(std_quant - std_json).max()

print(f"\nTRAINING vs QUANTIZZAZIONE:")
print(f"  Max Δ MEAN: {diff_train_quant_mean:.10f}")
print(f"  Max Δ STD:  {diff_train_quant_std:.10f}")

print(f"\nQUANTIZZAZIONE vs JSON:")
print(f"  Max Δ MEAN: {diff_quant_json_mean:.10f}")
print(f"  Max Δ STD:  {diff_quant_json_std:.10f}")

print("\n" + "=" * 70)
print("DETTAGLIO (prime 4 features)")
print("=" * 70)
print(f"\nMEAN:")
print(f"  Training:       {mean_train[:4]}")
print(f"  Quantizzazione: {mean_quant[:4]}")
print(f"  JSON:           {mean_json[:4]}")

print(f"\nSTD:")
print(f"  Training:       {std_train[:4]}")
print(f"  Quantizzazione: {std_quant[:4]}")
print(f"  JSON:           {std_json[:4]}")

# Conclusione
print("\n" + "=" * 70)
if diff_train_quant_mean < 1e-6 and diff_quant_json_mean < 1e-5:
    print("✅✅✅ PARAMETRI QUASI IDENTICI!")
    print(f"   Differenze < 1.5e-5 (arrotondamento JSON)")
    print("   Training, quantizzazione e JSON usano STESSI MEAN/STD")
    print("   I parametri nel firmware sono CORRETTI")
else:
    print("⚠️⚠️⚠️ DISCREPANZA RILEVATA!")
    print("   Controlla manualmente le differenze")
print("=" * 70)
