#!/usr/bin/env python3
"""
Genera finestra RAW (non standardizzata) per testare il firmware
che ora applica la standardizzazione on-board
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_PATH = "/home/cardigun/Scrivania/Progetto SAFE/NOW/data/processed_streaming_row_continuous.csv"
NFEAT = 16
WINDOW = 30

print("=" * 70)
print("GENERAZIONE FINESTRA RAW PER TEST FIRMWARE")
print("=" * 70)

# Carica dataset
df = pd.read_csv(CSV_PATH, header=0, sep=",", low_memory=False)
num_df = df.select_dtypes(include=[np.number])
data = num_df.iloc[:, :NFEAT].to_numpy(dtype=np.float32)

print(f"\nDataset: {data.shape[0]} righe x {data.shape[1]} features")

# Prendi prima finestra RAW
win_raw = data[:WINDOW]

# Calcola scaler (come nel training) per confronto
scaler = StandardScaler()
scaler.fit(data)
win_std = scaler.transform(win_raw)

# Salva entrambe
np.savetxt("win_raw.csv", win_raw, delimiter=",", fmt="%.6f")
np.savetxt("win_std_reference.csv", win_std, delimiter=",", fmt="%.6f")

print("\n" + "=" * 70)
print("FILE GENERATI")
print("=" * 70)
print("\n1. win_raw.csv")
print("   → Da inviare al firmware (30 righe x 16 valori RAW)")
print("   → Il firmware applicherà standardizzazione on-board")
print(f"   → Range: [{win_raw.min():.3f}, {win_raw.max():.3f}]")

print("\n2. win_std_reference.csv")
print("   → Versione già standardizzata (per confronto)")
print("   → L'output del modello deve essere UGUALE")
print(f"   → Range: [{win_std.min():.3f}, {win_std.max():.3f}]")

print("\n" + "=" * 70)
print("TEST PROCEDURE")
print("=" * 70)
print("\nOPZIONE A: Test con firmware vecchio (pre-standardizzato)")
print("  python3 invio.py   # usa win_std_reference.csv")
print("\nOPZIONE B: Test con firmware nuovo (standardizza on-board)")
print("  1. python3 switch_to_raw.py  # modifica invio.py")
print("  2. python3 invio.py")
print("  3. Confronta output: devono essere IDENTICI")

print("\n" + "=" * 70)
print("VERIFICA PARAMETRI")
print("=" * 70)
print(f"\nMean[0..3]: {scaler.mean_[:4]}")
print(f"Std[0..3]:  {scaler.scale_[:4]}")
print("\n✅ Controlla che questi valori corrispondano a quelli nel firmware!")

print("\n" + "=" * 70)
print("PRIMA RIGA ESEMPIO")
print("=" * 70)
print(f"\nRAW:  {win_raw[0][:4]}...")
print(f"STD:  {win_std[0][:4]}...")
print(f"\nFormula applicata: z = (x - mean) / std")
print(f"  Feature 0: ({win_raw[0][0]:.6f} - {scaler.mean_[0]:.6f}) / {scaler.scale_[0]:.6f} = {win_std[0][0]:.6f}")

print("\n" + "=" * 70)
print("✅ FATTO!")
print("=" * 70)
