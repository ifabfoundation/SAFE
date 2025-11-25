#!/usr/bin/env python3
"""
Confronta output inferenza STM32 vs Keras/TFLite.
Calcola metriche di accuratezza: MAE, MSE, RMSE, Max Diff.
"""

import json
import numpy as np
import sys
from pathlib import Path


def load_json(path):
    """Carica file JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def compare_outputs(stm32_outputs, keras_outputs, tolerance=1e-3):
    """
    Confronta output STM32 vs Keras.
    
    Args:
        stm32_outputs: lista di float (da STM32)
        keras_outputs: lista di float (da Keras/TFLite)
        tolerance: tolleranza per considerare valori "uguali"
    
    Returns:
        dict con metriche di confronto
    """
    stm32 = np.array(stm32_outputs, dtype=np.float32)
    keras = np.array(keras_outputs, dtype=np.float32)
    
    if len(stm32) != len(keras):
        print(f"[WARN] Lunghezze diverse: STM32={len(stm32)}, Keras={len(keras)}")
        # Tronca al minimo
        min_len = min(len(stm32), len(keras))
        stm32 = stm32[:min_len]
        keras = keras[:min_len]
    
    # Calcola differenze
    diff = np.abs(stm32 - keras)
    
    # Metriche
    mae = np.mean(diff)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_diff = np.max(diff)
    max_diff_idx = np.argmax(diff)
    
    # Conta valori entro tolleranza
    within_tolerance = np.sum(diff <= tolerance)
    percent_ok = (within_tolerance / len(diff)) * 100
    
    metrics = {
        'count': len(stm32),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'max_diff': float(max_diff),
        'max_diff_index': int(max_diff_idx),
        'max_diff_stm32': float(stm32[max_diff_idx]),
        'max_diff_keras': float(keras[max_diff_idx]),
        'within_tolerance': int(within_tolerance),
        'percent_within_tolerance': float(percent_ok),
        'tolerance': tolerance
    }
    
    return metrics, stm32, keras, diff


def print_comparison_table(stm32, keras, diff, max_rows=20):
    """Stampa tabella confronto primi N valori."""
    print("\n" + "=" * 80)
    print(f"{'Index':<8} {'STM32':<15} {'Keras':<15} {'Diff':<15} {'Status':<10}")
    print("=" * 80)
    
    for i in range(min(max_rows, len(stm32))):
        status = "✅ OK" if diff[i] <= 1e-3 else "⚠️  DIFF"
        print(f"{i:<8} {stm32[i]:<15.6f} {keras[i]:<15.6f} {diff[i]:<15.6e} {status:<10}")
    
    if len(stm32) > max_rows:
        print(f"... ({len(stm32) - max_rows} righe omesse)")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Confronta output STM32 vs Keras')
    parser.add_argument('--stm32', required=True, help='File JSON output STM32 (da capture_stm32_output.py)')
    parser.add_argument('--keras', required=True, help='File JSON output Keras (da keras_inference.py)')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Tolleranza per match (default 0.001)')
    parser.add_argument('--output', default='comparison_report.json', help='File report JSON')
    args = parser.parse_args()
    
    # Carica risultati
    print(f"[*] Caricamento STM32 output: {args.stm32}")
    stm32_data = load_json(args.stm32)
    
    print(f"[*] Caricamento Keras output: {args.keras}")
    keras_data = load_json(args.keras)
    
    # Verifica presenza inferenze STM32
    if not stm32_data.get('inferences'):
        print("[ERROR] Nessuna inferenza trovata in STM32 output!")
        print("[HINT] Assicurati di aver eseguito capture_stm32_output.py durante test")
        sys.exit(1)
    
    # Prendi prima inferenza STM32
    stm32_inference = stm32_data['inferences'][0]
    stm32_outputs = stm32_inference['outputs']
    
    print(f"\n[INFO] STM32: {len(stm32_data['inferences'])} inferenze catturate")
    print(f"[INFO] STM32 inference #1: {len(stm32_outputs)} output values")
    print(f"[INFO] Keras: {keras_data['output_count']} output values")
    
    # Confronta
    print("\n[*] Confronto in corso...")
    metrics, stm32_arr, keras_arr, diff_arr = compare_outputs(
        stm32_outputs, 
        keras_data['outputs'],
        args.tolerance
    )
    
    # Stampa metriche
    print("\n" + "=" * 80)
    print("METRICHE DI CONFRONTO STM32 vs KERAS")
    print("=" * 80)
    print(f"Numero valori:              {metrics['count']}")
    print(f"MAE (Mean Absolute Error):  {metrics['mae']:.6e}")
    print(f"MSE (Mean Squared Error):   {metrics['mse']:.6e}")
    print(f"RMSE:                       {metrics['rmse']:.6e}")
    print(f"Max Difference:             {metrics['max_diff']:.6e}")
    print(f"  @ index {metrics['max_diff_index']}: STM32={metrics['max_diff_stm32']:.6f}, Keras={metrics['max_diff_keras']:.6f}")
    print(f"\nValori entro tolleranza ({args.tolerance}):")
    print(f"  {metrics['within_tolerance']} / {metrics['count']} ({metrics['percent_within_tolerance']:.2f}%)")
    
    # Valutazione qualitativa
    if metrics['max_diff'] < 1e-4:
        verdict = "✅ ECCELLENTE - Differenza trascurabile (quantizzazione match perfetto)"
    elif metrics['max_diff'] < 1e-3:
        verdict = "✅ OTTIMO - Differenza accettabile per INT8 quantizzato"
    elif metrics['max_diff'] < 1e-2:
        verdict = "⚠️  ACCETTABILE - Differenza visibile ma entro limiti quantizzazione"
    else:
        verdict = "❌ PROBLEMA - Differenza significativa, verificare configurazione"
    
    print(f"\n{verdict}")
    print("=" * 80)
    
    # Tabella confronto
    print_comparison_table(stm32_arr, keras_arr, diff_arr, max_rows=20)
    
    # Salva report
    report = {
        'stm32_file': args.stm32,
        'keras_file': args.keras,
        'stm32_frames_received': stm32_data.get('frames_received', 0),
        'stm32_inferences_count': len(stm32_data['inferences']),
        'metrics': metrics,
        'verdict': verdict,
        'tolerance': args.tolerance
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Report salvato in: {args.output}")
    
    # Exit code per CI
    if metrics['max_diff'] > 1e-2:
        print("\n[FAIL] Test fallito: differenza troppo grande")
        sys.exit(1)
    else:
        print("\n[PASS] Test superato ✅")
        sys.exit(0)


if __name__ == '__main__':
    main()
