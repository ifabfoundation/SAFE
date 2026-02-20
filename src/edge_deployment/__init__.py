"""
Edge Deployment Module
======================
Complete pipeline from trained model to microcontroller deployment:
- INT8 quantization (representative dataset, TFLite conversion, benchmarking)
- STM32U545RE firmware (X-CUBE-AI, UART, Modbus RTU over RS485)
- ESP32-WROVER porting (SECO EasyEdge, ESP-IDF + TFLite Micro)
- Validation (PC vs. MCU inference comparison, standardization verification)

Contributors: Francesco Simoni (Università di Bologna)
Coordinators: Orso Peruzzi, Benedetta Baldini
"""
