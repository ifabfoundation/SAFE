# SAFE — Anomaly Detection (ARM/RISC-V ready)

This repository collects everything built so far to train, quantize, benchmark, and embed a multivariate time-series anomaly-detection model on microcontrollers It includes data exploration, model training/export, representative-dataset generation for INT8 quantization, automated sweeps for benchmarking, and STM32CubeIDE/X-CUBE-AI projects to run inference on target (UART + Modbus RTU, with and without onboard standardization). Next step is ESP32 Porting.

## Repository layout (what’s where)

```
SAFE
├── 06 francesco.tex
├── README.md
├── Analysis/
│   ├── analisi.py
│   ├── stats_completo.csv
│   └── top_corr_pairs.csv
├── Training-Export/
│   ├── cnn_lstm.py
│   └── export_from_h5.py
├── Pipeline/
│   ├── bench_tflite.py
│   ├── convert_savedmodel.py
│   ├── make_rep_balanced_commented.py
│   ├── sweep_quant_bench.py
│   └── qbench_sweep_20251027_165731/
│       ├── config.json
│       └── results.csv
├── STM32 Interfacing/
│   ├── app_x-cube-ai.c
│   ├── invio.py
│   ├── main.c
│   └── Prova.ioc
├── Benchmarking/
│   ├── batch_golden_summary.py
│   ├── crea_finestra_std.py
│   ├── golden_compare_all.py
│   ├── results_golden_aug.csv
│   ├── 2000-05/
│   │   ├── compare_human.csv
│   │   └── outINT8.txt
│   └── 500-06/
│       ├── compare_human.csv
│       └── outINT8.txt
├── Benchmarking - 2/
│   ├── batch_golden_summary.py
│   ├── bench_tflite.py
│   ├── convert_savedmodel.py
│   ├── crea_finestra_std.py
│   ├── golden_compare_all.py
│   ├── keras_vs_tflite.py
│   ├── make_rep_balanced_commented.py
│   ├── results_golden_aug.csv
│   └── sweep_quant_bench.py
├── Serial/
│   ├── app_x-cube-ai.c
│   ├── main.c
│   └── Prova.ioc
├── OnboardStandardization/
│   ├── generate_raw_window.py
│   ├── invio.py
│   ├── switch_to_raw.py
│   ├── verify_all_scalers.py
│   └── verify_standardization.py
├── ModBus/
│   ├── app_x-cube-ai.c
│   ├── main.c
│   ├── modbus_rtu.c
│   ├── modbus_rtu.h
│   └── Prova.ioc
├── Inference-Test/
│   ├── capture_stm32_output.py
│   ├── compare_inference.py
│   └── keras_inference.py
└── Esp32/
```

### Folder overview

* **06 francesco.tex** – Mine LaTeX chapter describing the full pipeline:
  training → quantization → benchmarking → STM32 integration (UART + Modbus RTU).

* **Analysis/**
  Quick EDA tools used in *Step 2 – data analysis* of the pipeline:

  * `analisi.py`: prints histograms, correlation matrix, and exports CSV summaries.
  * `stats_completo.csv`: global statistics (mean, std, min, max, etc.).
  * `top_corr_pairs.csv`: most correlated variable pairs (used to reason about quantization and tails).

* **Training-Export/**
  *Step 1 – training realistic model*:

  * `cnn_lstm.py`: trains the compact CNN+LSTM forecaster (window=30, forecast=10, 16 features) and exports a Keras `SavedModel` (unrolled, ready for TFLite).
  * `export_from_h5.py`: helper to re-export an existing `.h5` model as `SavedModel` if needed.

* **Pipeline/**
  Core quantization pipeline from the thesis *Section 5 – Quantization*:

  * `make_rep_balanced_commented.py`: builds `rep_windows_balanced.npz` from the processed CSV:
    cleans non-numeric columns, standardizes, scores windows, and selects a balanced mix
    of “normal” and tail (high-score) windows.
  * `convert_savedmodel.py`: converts the float32 `SavedModel` to full-INT8 `.tflite` using the representative dataset.
  * `bench_tflite.py`: compares Keras vs TFLite (MSE/MAE/RMSE/MAX, latency) on the representative windows.
  * `sweep_quant_bench.py`: runs a grid search over `(max_total × balance)` and logs each config to a `qbench_sweep_*` folder.
  * Example sweep:

    * `qbench_sweep_20251027_165731/config.json`: parameters (max_totals, balances, etc.).
    * `qbench_sweep_20251027_165731/results.csv`: one row per configuration.

* **Benchmarking/**
  First “golden window” experiments (Keras vs TFLite vs STM32) used in the chapter:

  * `crea_finestra_std.py`: generates a standardized benchmark window from the processed CSV.
  * `golden_compare_all.py`: runs a three-way comparison (Keras, TFLite, STM32) and exports CSV summaries.
  * `batch_golden_summary.py`: aggregates multiple golden runs.
  * `results_golden_aug.csv`: augmented metrics for the different golden tests.
  * `2000-05/`, `500-06/`: per-experiment folders with:

    * `compare_human.csv`: human-readable summary (per-element deltas and aggregate metrics).
    * `outINT8.txt`: raw INT8 outputs captured from the MCU.

* **Benchmarking - 2/**
  Cleaned-up / refactored benchmarking code:

  * Same idea as `Benchmarking/`, but organized for re-use and automation.
  * `keras_vs_tflite.py`: direct comparison of PC Keras vs PC TFLite on windows.
  * `sweep_quant_bench.py`, `bench_tflite.py`, `make_rep_balanced_commented.py`: “final” scripts matching the thesis description.
  * `results_golden_aug.csv`: consolidated results for the updated pipeline.

* **STM32 Interfacing/**
  First STM32CubeIDE / X-CUBE-AI project used to validate basic deployment on NUCLEO boards:

  * `Prova.ioc`: CubeMX project file.
  * `app_x-cube-ai.c`, `main.c`: integration of the `.tflite` network, float/logging options, and UART hooks.
  * `invio.py`: Python sender script for test windows via UART (ASCII).

* **Serial/**
  Minimal UART project focused on serial communication only:

  * Same structure (`Prova.ioc`, `app_x-cube-ai.c`, `main.c`), but streamlined for 1-Hz acquisition and log over Virtual COM.
  * Used to test STM32 vs TFLite consistency in the *serial* setting.

* **OnboardStandardization/**
  Moves standardization **onto the MCU**, avoiding loss of precision when exporting pre-standardized windows:

  * `generate_raw_window.py`: extracts raw (non-standardized) windows from the CSV.
  * `switch_to_raw.py`: helpers to migrate existing experiments from standardized to raw + on-board standardization.
  * `verify_standardization.py`, `verify_all_scalers.py`: checks that STM32’s µ/σ and the PC’s
    `StandardScaler` produce numerically equivalent outputs.
  * `invio.py`: UART sender for the raw window pipeline.

* **ModBus/**
  Final **Modbus RTU over RS-485** implementation from *Section 7 – Modbus RTU Implementation*:

  * `modbus_rtu.c`, `modbus_rtu.h`: Modbus RTU state machine, frame parsing, CRC, and register mapping for the anomaly score.
  * `app_x-cube-ai.c`, `main.c`: integrated X-CUBE-AI network with Modbus RTU I/O instead of simple UART logging.
  * `Prova.ioc`: CubeMX project with USART/RS-485 configuration (DE/RE, baudrate, etc.).
  * Uses the same Python tooling (`invio.py`, golden windows) to send test sequences, now wrapped in Modbus frames.

* **Inference-Test/**
  PC-side tooling for *end-to-end* comparisons:

  * `keras_inference.py`: runs inference with the Keras model on the test windows.
  * `capture_stm32_output.py`: records raw STM32 outputs from the serial interface (UART / RS-485 bridge).
  * `compare_inference.py`: aligns PC vs MCU outputs and computes deltas (MSE, RMSE, MAE, MAX).

* **Esp32/**
  Scratch/experimental area for porting the same `.tflite` model and pipeline ideas
  to ESP32 / other ARM/RISC-V targets. At the moment it is not a polished project.

## What’s been done so far (short version)

* **Data exploration (EDA).** Produced global stats, correlations, and distributions to understand stability, tails, and cross-sensor relations used later to choose quantization strategy and features. Artifacts are under `Analysis/` (`stats_completo.csv`, `top_corr_pairs.csv`, plus the plotting script in `analisi.py`).

* **Baseline model.** Implemented a **compact CNN+LSTM** multivariate forecaster (window=30, forecast=10) trained on standardized features; exported as a Keras `SavedModel` (`export_cnn_lstm/savedmodel_unroll`) ready for conversion.

* **Representative dataset (INT8).** Built a robust tool (`Pipeline/make_rep_balanced_commented.py`) that:

  1. cleans non-numeric/timestamp columns,
  2. standardizes,
  3. **scores windows** (optionally passing through the SavedModel),
  4. **selects a balanced mix of “normal” and high-score “tail” windows**,
  5. saves `rep_windows_balanced.npz` + CSV summaries.

* **Post-training quantization to INT8** and **benchmarking.** Converted to full-INT8 TFLite using the representative set, then measured Keras vs TFLite deltas (MSE/MAE/RMSE/MAX) and latency. The **sweep runner** automates a grid over `(max_total × balance)`, logging artifacts and a single `results.csv` per run.

* **MCU bring-up (STM32, UART + Modbus RTU).** Created multiple CubeIDE projects (U5/F4) with X-CUBE-AI that ingest a `.tflite`, set up UART/Timer or RS-485 / Modbus RTU, and run the network loop. The code base covers:

  * classic UART logging,
  * **on-board standardization** of raw windows,
  * industrial **Modbus RTU** framing on RS-485.

## Quick start (minimal commands)

### 0) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### 1) Train & export baseline (float32)

```bash
python3 Training-Export/cnn_lstm.py
# → exports export_cnn_lstm/savedmodel_unroll (Keras SavedModel for the CNN+LSTM)
```

### 2) Build representative dataset (balanced tails)

```bash
python3 Pipeline/make_rep_balanced_commented.py \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --outdir Pipeline/rep_2000_b0p5 \
  --window 30 --forecast 10 --nfeat 16 \
  --max_total 2000 --balance 0.5 --seed 42
```

### 3) Convert to full-INT8 TFLite

```bash
python3 Pipeline/convert_savedmodel.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --outdir Pipeline/tflite_2000_b0p5
```

### 4) Benchmark single run (Keras ↔ TFLite)

```bash
python3 Pipeline/bench_tflite.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --tflite Pipeline/tflite_2000_b0p5/model_int8_full.tflite \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --n 256 --threads 1
```

### 5) Run a full sweep – many configurations

```bash
python3 Pipeline/sweep_quant_bench.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --window 30 --forecast 10 --nfeat 16 \
  --max-totals 500,1000,2000,3000,4000,5000,6000 \
  --balances 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --threads 1 --n-rep 256
```

Each sweep run creates its own experiment folder under `Pipeline/qbench_sweep_…` with `results.csv` and subfolders per configuration.

### 6) Re-run the “golden window” benchmark

From inside `Benchmarking/` (or `Benchmarking - 2/`):

```bash
python3 golden_compare_all.py \
  --savedmodel ../export_cnn_lstm/savedmodel_unroll \
  --tflite ../Pipeline/tflite_2000_b0p5/model_int8_full.tflite \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --start 0
```

This regenerates:

* raw INT8 outputs (`outINT8.txt`) from the MCU,
* human-readable summaries (`compare_human.csv`),
* aggregated CSVs (`results_golden_aug.csv`).



## Key results so far (from the internal report)

A robust setting used for deployment trials is:

* **Representative selection:** `max_total = 2000`, `balance = 0.5`.

* **Keras vs TFLite (test bench, standardized windows):**

  * MSE ≈ 0.14
  * MAE ≈ 0.28
  * MAX ≈ 1.17
    (RMSE can be derived from MSE and is ≈ 0.37.)

* **On-device (STM32) vs TFLite (raw windows + onboard standardization):**

  [
  \begin{aligned}
  \Delta\text{MSE} &\approx +0.0079 \
  \Delta\text{RMSE} &\approx +0.0110 \
  \Delta\text{MAE} &\approx +0.0022 \
  \Delta\text{MAX} &\approx +0.0517
  \end{aligned}
  ]

  which corresponds to **ΔMAX ≈ 1 LSB** when the output scale
  is about 0.05173 (as discussed in the chapter).


