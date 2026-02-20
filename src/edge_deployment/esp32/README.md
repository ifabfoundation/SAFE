# ESP32 SECO EasyEdge Deployment

This module contains the firmware adaptation for deploying the SAFE anomaly detection
model on the ESP32-WROVER (SECO EasyEdge board) using TFLite Micro.

## Status

The ESP32 deployment was developed on a separate SECO development environment.
See Section 6.6 of the scientific report (`docs/content/06_edge_deployment.tex`)
for implementation details, including:

- ESP-IDF project setup and TFLite Micro integration
- Model conversion from STM32 X-CUBE-AI to TFLite Micro format
- UART-based inference pipeline (Float32)
- Known issues (WHILE op support via `esp-tflite-micro`)

## Hardware

- **Board:** ESP32-WROVER 3.3V (SECO EasyEdge)
- **Framework:** ESP-IDF + TFLite Micro
- **Communication:** Serial USB (debug) / Modbus RTU (production)
