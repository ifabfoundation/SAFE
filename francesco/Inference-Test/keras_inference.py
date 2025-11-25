#!/usr/bin/env python3
"""
Esegue inferenza Keras/TFLite sugli stessi dati di test per confronto con STM32.
Usa il modello mt500_b0p6 INT8 quantizzato.
"""

import numpy as np
import tensorflow as tf
import json
import sys
from pathlib import Path

# MEAN e STD per standardizzazione (stessi valori del firmware!)
MEAN = np.array([
    3.79418182, -0.09481818, 0.83072727, 0.41945455, 1.68581818,
    3.82218182, -0.09472727, 0.83027273, 0.41927273, 1.68581818,
    3.82145455, -0.09481818, 0.83036364, 0.41936364, 1.68581818,
    52.5
], dtype=np.float32)

STD = np.array([
    0.53918436, 0.42606455, 0.13408624, 0.03127939, 0.01916632,
    0.53832074, 0.42606455, 0.13397518, 0.03114134, 0.01916632,
    0.53848584, 0.42606455, 0.13400142, 0.03117635, 0.01916632,
    0.0  # STD[15] = 0 → feature costante
], dtype=np.float32)


def load_win_raw(csv_path):
    """Carica win_raw.csv (30 righe × 16 colonne)."""
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    print(f"[OK] Caricato {csv_path}: shape {data.shape}")
    return data


def standardize_window(window_raw):
    """
    Standardizza finestra 30×16 con z-score (come firmware STM32).
    
    Args:
        window_raw: numpy array (30, 16)
    
    Returns:
        window_std: numpy array (30, 16) standardizzato
    """
    window_std = np.zeros_like(window_raw)
    for i in range(16):
        if STD[i] != 0.0:
            window_std[:, i] = (window_raw[:, i] - MEAN[i]) / STD[i]
        else:
            window_std[:, i] = 0.0  # Evita divisione per zero
    
    return window_std


def run_tflite_inference(model_path, input_data):
    """
    Esegue inferenza con TFLite INT8 quantizzato.
    
    Args:
        model_path: path al file .tflite
        input_data: numpy array (1, 30, 16, 1) float32 GIÀ STANDARDIZZATO
    
    Returns:
        output: numpy array dei risultati (dopo dequantizzazione)
    """
    # Carica interpreter
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Input details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"[INFO] Input shape: {input_details['shape']}")
    print(f"[INFO] Input dtype: {input_details['dtype']}")
    print(f"[INFO] Output shape: {output_details['shape']}")
    print(f"[INFO] Output dtype: {output_details['dtype']}")
    
    # Quantizza input se necessario
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        print(f"[INFO] Input quantization: scale={input_scale}, zero_point={input_zero_point}")
        
        # Quantizza: int8 = round(float32 / scale) + zero_point
        input_quantized = np.round(input_data / input_scale).astype(np.int8) + input_zero_point
        input_quantized = np.clip(input_quantized, -128, 127).astype(np.int8)
    else:
        input_quantized = input_data.astype(input_details['dtype'])
    
    # Set input tensor
    interpreter.set_tensor(input_details['index'], input_quantized)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_quantized = interpreter.get_tensor(output_details['index'])
    
    # Dequantizza output se necessario
    if output_details['dtype'] == np.int8:
        output_scale, output_zero_point = output_details['quantization']
        print(f"[INFO] Output quantization: scale={output_scale}, zero_point={output_zero_point}")
        
        # Dequantizza: float32 = (int8 - zero_point) * scale
        output = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
    else:
        output = output_quantized
    
    return output.flatten()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inferenza Keras per confronto con STM32')
    parser.add_argument('--model', required=True, help='Path al modello .tflite (mt500_b0p6)')
    parser.add_argument('--data', required=True, help='Path a win_raw.csv')
    parser.add_argument('--output', default='keras_output.json', help='File output JSON')
    args = parser.parse_args()
    
    # Carica dati
    window_raw = load_win_raw(args.data)
    
    if window_raw.shape != (30, 16):
        print(f"[ERROR] win_raw.csv deve essere (30, 16), trovato {window_raw.shape}")
        sys.exit(1)
    
    # Standardizza (come STM32)
    print("[*] Standardizzazione z-score...")
    window_std = standardize_window(window_raw)
    print(f"[OK] Window standardizzata: min={window_std.min():.4f}, max={window_std.max():.4f}")
    
    # Reshape per TFLite: (1, 30, 16, 1)
    input_tensor = window_std.reshape(1, 30, 16, 1).astype(np.float32)
    
    # Inferenza
    print(f"[*] Caricamento modello: {args.model}")
    if not Path(args.model).exists():
        print(f"[ERROR] Modello non trovato: {args.model}")
        sys.exit(1)
    
    print("[*] Esecuzione inferenza TFLite...")
    output = run_tflite_inference(args.model, input_tensor)
    
    print(f"[OK] Inferenza completata: {len(output)} output values")
    print(f"[STATS] Output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"[STATS] Output mean: {output.mean():.6f}")
    print(f"[STATS] Output std: {output.std():.6f}")
    
    # Primi 10 valori per debug
    print("\n[DEBUG] Primi 10 output values:")
    for i in range(min(10, len(output))):
        print(f"  Output[{i:3d}] = {output[i]:.6f}")
    
    # Salva risultati
    result = {
        'model_path': args.model,
        'data_path': args.data,
        'window_shape': window_raw.shape,
        'output_count': len(output),
        'outputs': output.tolist(),
        'stats': {
            'min': float(output.min()),
            'max': float(output.max()),
            'mean': float(output.mean()),
            'std': float(output.std())
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[OK] Output salvato in: {args.output}")


if __name__ == '__main__':
    main()
