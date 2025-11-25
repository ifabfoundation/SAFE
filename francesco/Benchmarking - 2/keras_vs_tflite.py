#!/usr/bin/env python3
# compare_all.py
# Confronto Keras vs TFLite e Keras vs STM32 con dequant corretto e controlli robusti.


# python3 keras_vs_tflite.py \
#   --savedmodel ../export_cnn_lstm/savedmodel_unroll \
#   --tflite ../finale/qbench_sweep_20251027_165731/experiments/mt2000_b0p5/tflite/model_int8_full.tflite \
#   --rep     ../finale/qbench_sweep_20251027_165731/experiments/mt2000_b0p5/rep/rep_windows_balanced.npz \
#   --n 256 --threads 1 \
#   --stm32 outINT8.txt

import os, re, argparse, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
from keras.layers import TFSMLayer
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True, linewidth=140)

# ----------------------------- IO & preprocessing -----------------------------

def load_windows_from_npz(npz_path: str, n: Optional[int]=None) -> np.ndarray:
    data = np.load(npz_path)
    X = data['X'] if 'X' in data.files else list(data.values())[0]
    X = X.astype(np.float32, copy=False)
    if n is not None and len(X) > n:
        stride = max(1, len(X)//n)
        X = X[::stride][:n]
    return X  # [B,30,16]

def load_windows_from_csv(csv_path: str, window: int, nfeat: int, n: Optional[int]=None) -> np.ndarray:
    df = pd.read_csv(csv_path, low_memory=False)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < nfeat:
        nfeat = len(num_cols)
    data = df[num_cols[:nfeat]].to_numpy(dtype=np.float32, copy=False)
    del df

    # Standardizzazione (come nei tuoi script)
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data).astype(np.float32, copy=False)

    # Sliding windows senza forecast (servono solo input al modello)
    n_rows = data_std.shape[0]
    last_start = n_rows - window + 1
    starts = range(0, max(0, last_start))
    X = []
    for s in starts:
        X.append(data_std[s:s+window])
    if not X:
        raise RuntimeError("CSV troppo corto per generare almeno una finestra.")
    X = np.stack(X, axis=0).astype(np.float32)
    if n is not None and len(X) > n:
        stride = max(1, len(X)//n)
        X = X[::stride][:n]
    return X  # [B,30,16]

# ----------------------------- Model runners ----------------------------------

def run_keras_savedmodel(savedmodel_dir: str, X: np.ndarray) -> Tuple[np.ndarray, float]:
    layer = TFSMLayer(savedmodel_dir, call_endpoint='serving_default')
    t0 = time.time()
    y = layer(X, training=False)
    ms = (time.time() - t0) * 1000.0 / max(1, len(X))
    if isinstance(y, dict): y = next(iter(y.values()))
    elif isinstance(y, (list, tuple)): y = y[0]
    y = tf.convert_to_tensor(y).numpy().astype(np.float32, copy=False)
    return y, ms  # shape [B, 160] o [B,10,16]

def get_tflite_io_qparams(tfl_path: str) -> Tuple[Tuple[float,int], Tuple[float,int]]:
    interp = tf.lite.Interpreter(model_path=tfl_path)
    interp.allocate_tensors()
    i_det = interp.get_input_details()[0]
    o_det = interp.get_output_details()[0]
    in_scale, in_zp  = i_det.get('quantization', (None, None))
    out_scale, out_zp = o_det.get('quantization', (None, None))
    if in_scale is None or out_scale is None:
        raise RuntimeError("Modello TFLite senza quantization info.")
    return (float(in_scale), int(in_zp)), (float(out_scale), int(out_zp))

def run_tflite_float_out(tfl_path: str, X_float32: np.ndarray, threads: int=1) -> Tuple[np.ndarray, float]:
    """Quantizza input secondo (scale,zp) input e dequantizza output secondo quelli di output."""
    interp = tf.lite.Interpreter(model_path=tfl_path, num_threads=threads)
    interp.allocate_tensors()
    i_det = interp.get_input_details()[0]
    o_det = interp.get_output_details()[0]
    in_scale, in_zp  = i_det['quantization']
    out_scale, out_zp = o_det['quantization']

    Xq = np.round(X_float32 / in_scale + in_zp).astype(np.int8)
    Xq = np.clip(Xq, -128, 127)

    Yf_all = []
    t0 = time.time()
    for i in range(len(Xq)):
        interp.set_tensor(i_det['index'], Xq[i:i+1])
        interp.invoke()
        Yq = interp.get_tensor(o_det['index']).astype(np.int8)
        Yf = (Yq.astype(np.float32) - out_zp) * out_scale
        Yf_all.append(Yf)
    ms = (time.time() - t0) * 1000.0 / len(Xq)
    Yf_all = np.concatenate(Yf_all, axis=0).astype(np.float32)
    return Yf_all, ms

# ----------------------------- STM32 helpers ----------------------------------

def read_numeric_file(path: str) -> np.ndarray:
    txt = Path(path).read_text(encoding='utf-8', errors='ignore')
    # cattura sia interi che float con segno
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", txt)
    if not nums:
        raise RuntimeError(f"Nessun numero trovato in {path}")
    arr = np.array([float(x) for x in nums], dtype=np.float32)
    return arr

def dequant_if_int8(arr: np.ndarray, out_scale: float, out_zp: int) -> Tuple[np.ndarray, str]:
    """Tenta di capire se l'array è int8 grezzo (-128..127) o già float."""
    # euristica: percentuale di valori interi nel range int8
    ints = np.isclose(arr, np.round(arr))
    within = (arr >= -128) & (arr <= 127)
    frac_like_int8 = float(np.mean(ints & within)) if arr.size else 0.0

    if frac_like_int8 > 0.90:
        yq = np.round(arr).astype(np.int8)
        y = (yq.astype(np.float32) - out_zp) * out_scale
        return y.astype(np.float32), "int8→float(dequant)"
    else:
        return arr.astype(np.float32), "float(pass-through)"

# ----------------------------- Metrics & utils --------------------------------

def flatten_last(y: np.ndarray) -> np.ndarray:
    # uniforma: [B,160] oppure [B,10,16] → [B,160]
    y = np.array(y)
    if y.ndim == 3:
        b,t,f = y.shape
        return y.reshape(b, t*f)
    return y

def metrics(y_ref: np.ndarray, y_hat: np.ndarray) -> Tuple[float,float,float]:
    diff = y_hat - y_ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    maxe = float(np.max(np.abs(diff)))
    return mse, mae, maxe

def brief_range(y: np.ndarray) -> str:
    return f"[{np.min(y):.3f}, {np.max(y):.3f}]"

# ---------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Confronto Keras↔TFLite e Keras↔STM32 (con dequant automatico).")
    ap.add_argument("--savedmodel", required=True)
    ap.add_argument("--tflite", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rep", help="Representative NPZ (preferito)")
    g.add_argument("--csv", help="CSV grezzo per generare finestre")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--nfeat", type=int, default=16)
    ap.add_argument("--n", type=int, default=256, help="numero max finestre da testare")
    ap.add_argument("--threads", type=int, default=1)

    ap.add_argument("--stm32", help="File output STM32 (int8 o float)")
    ap.add_argument("--dump_pairs", type=int, default=10, help="quante coppie stampare a video")
    args = ap.parse_args()

    # Carica input X
    if args.rep:
        X = load_windows_from_npz(args.rep, n=args.n)
    else:
        X = load_windows_from_csv(args.csv, window=args.window, nfeat=args.nfeat, n=args.n)

    if X.ndim != 3 or X.shape[1] != args.window:
        raise RuntimeError(f"Input X con shape inattesa: {X.shape} (atteso [B,{args.window},{args.nfeat}])")

    print(f"[INFO] Sample X: {X.shape} ({X.dtype})")

    # Keras
    yk, ms_k = run_keras_savedmodel(args.savedmodel, X)
    yk = flatten_last(yk)
    print(f"[KERAS] out: {yk.shape}  range={brief_range(yk)}  ~{ms_k:.3f} ms/finestra")

    # TFLite
    (in_scale, in_zp), (out_scale, out_zp) = get_tflite_io_qparams(args.tflite)
    print(f"[TFLITE] qparams: IN (scale={in_scale:.9f}, zp={in_zp})  OUT (scale={out_scale:.9f}, zp={out_zp})")

    yt, ms_t = run_tflite_float_out(args.tflite, X, threads=args.threads)
    yt = flatten_last(yt)
    print(f"[TFLITE] out: {yt.shape}  range={brief_range(yt)}  ~{ms_t:.3f} ms/finestra")

    mse_t, mae_t, maxe_t = metrics(yk, yt)
    print(f"\nΔ TFLite vs Keras  →  MSE={mse_t:.6e}  MAE={mae_t:.6e}  MAX={maxe_t:.6e}")

    # STM32 (opzionale)
    if args.stm32:
        arr = read_numeric_file(args.stm32)
        # allow both [160] e [B*160]
        if arr.size % 160 != 0:
            print(f"[STM32] WARN: {arr.size} valori non multipli di 160, confronto sui primi 160.")
        L = min(yk.size, arr.size // (yk.shape[1] if yk.ndim==2 else 160) * 160)
        if L == 0:
            raise RuntimeError("STM32 file privo di sufficienti valori per il confronto.")

        ystm_raw = arr[:L].reshape(-1, 160)
        ystm, mode = dequant_if_int8(ystm_raw, out_scale, out_zp)
        print(f"[STM32] interpretazione: {mode}  out: {ystm.shape}  range={brief_range(ystm)}")

        # allinea alle prime B finestre usate da Keras
        B = min(yk.shape[0], ystm.shape[0])
        yk_b = yk[:B]
        ystm_b = ystm[:B]

        mse_s, mae_s, maxe_s = metrics(yk_b, ystm_b)
        print(f"Δ STM32 vs Keras   →  MSE={mse_s:.6e}  MAE={mae_s:.6e}  MAX={maxe_s:.6e}")

        # stampa coppie
        n_pairs = min(args.dump_pairs, yk_b.shape[1])
        print("\nPrime coppie (Keras | STM32):")
        for i in range(n_pairs):
            print(f"{yk_b[0,i]:8.3f} | {ystm_b[0,i]:8.3f}")

    # stampa coppie Keras vs TFLite
    n_pairs = min(args.dump_pairs, yk.shape[1])
    print("\nPrime coppie (Keras | TFLite):")
    for i in range(n_pairs):
        print(f"{yk[0,i]:8.3f} | {yt[0,i]:8.3f}")

    # riepilogo JSON (utile per parsing automatico)
    out = {
        "keras_ms": ms_k,
        "tflite_ms": ms_t,
        "qparams": {"in_scale": in_scale, "in_zp": in_zp, "out_scale": out_scale, "out_zp": out_zp},
        "delta_tflite": {"mse": mse_t, "mae": mae_t, "max": maxe_t},
    }
    if args.stm32:
        out["delta_stm32"] = {"mse": mse_s, "mae": mae_s, "max": maxe_s}
    print("\nJSON_SUMMARY:", json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    main()
