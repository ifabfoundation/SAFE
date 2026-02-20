#!/usr/bin/env python3
"""
Cattura output STM32 da /dev/ttyACM0 durante test con modbus_simulator.
Estrae i valori di inferenza e li salva per confronto con Keras.
"""

import serial
import re
import sys
import json
from datetime import datetime

def parse_stm32_output(port='/dev/ttyACM0', baud=115200, timeout_sec=120):
    """
    Cattura output STM32 e estrae inference results.
    
    Returns:
        dict: {
            'raw_output': str,
            'inferences': [
                {'frame_count': int, 'outputs': [float, ...], 'timestamp': str},
                ...
            ],
            'frames_received': int
        }
    """
    
    print(f"[*] Apertura porta {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"[ERROR] Impossibile aprire {port}: {e}")
        sys.exit(1)
    
    print("[OK] Porta aperta. Attendendo output STM32...")
    print("[INFO] Premi Ctrl+C per terminare\n")
    print("=" * 70)
    
    raw_lines = []
    inferences = []
    frames_received = 0
    
    # Pattern per rilevare frame Modbus OK
    frame_pattern = re.compile(r'\[MODBUS\] Frame OK #(\d+)')
    
    # Pattern per rilevare output inferenza
    # Es: "[AI] Output[0] = 0.052" oppure righe con valori numerici
    inference_start_pattern = re.compile(r'\[AI\].*INFERENZA|Inference done|Output')
    value_pattern = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
    
    current_inference = None
    collecting_values = False
    
    try:
        start_time = datetime.now()
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                
                raw_lines.append(line)
                print(line)
                
                # Conta frame ricevuti
                match = frame_pattern.search(line)
                if match:
                    frames_received = int(match.group(1))
                
                # Rileva inizio inferenza
                if inference_start_pattern.search(line):
                    if current_inference is not None and current_inference['outputs']:
                        inferences.append(current_inference)
                    
                    current_inference = {
                        'frame_count': frames_received,
                        'outputs': [],
                        'timestamp': datetime.now().isoformat()
                    }
                    collecting_values = True
                
                # Estrai valori numerici durante inferenza
                if collecting_values and current_inference is not None:
                    # Cerca pattern tipo "Output[0] = 0.052" o linee con numeri
                    values = value_pattern.findall(line)
                    if values:
                        current_inference['outputs'].extend([float(v) for v in values])
                    
                    # Fine inferenza quando vedi riga vuota o nuovo frame
                    if line.startswith('[MODBUS]') or line.startswith('[DBG]'):
                        if len(current_inference['outputs']) > 0:
                            inferences.append(current_inference)
                            current_inference = None
                            collecting_values = False
            
            # Timeout check
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_sec:
                print(f"\n[WARN] Timeout {timeout_sec}s raggiunto")
                break
                
    except KeyboardInterrupt:
        print("\n[*] Cattura interrotta da utente")
    finally:
        ser.close()
    
    # Salva ultimo inference se presente
    if current_inference is not None and current_inference['outputs']:
        inferences.append(current_inference)
    
    result = {
        'raw_output': '\n'.join(raw_lines),
        'inferences': inferences,
        'frames_received': frames_received,
        'capture_time': datetime.now().isoformat()
    }
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cattura output inferenza STM32')
    parser.add_argument('--port', default='/dev/ttyACM0', help='Porta seriale STM32')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in secondi')
    parser.add_argument('--output', default='stm32_output.json', help='File output JSON')
    args = parser.parse_args()
    
    result = parse_stm32_output(args.port, args.baud, args.timeout)
    
    print("\n" + "=" * 70)
    print(f"[STATS] Frame ricevuti: {result['frames_received']}")
    print(f"[STATS] Inferenze catturate: {len(result['inferences'])}")
    
    for i, inf in enumerate(result['inferences']):
        print(f"  Inferenza #{i+1}: {len(inf['outputs'])} output values")
    
    # Salva JSON
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[OK] Output salvato in: {args.output}")
    
    # Salva anche raw log
    raw_file = args.output.replace('.json', '_raw.txt')
    with open(raw_file, 'w') as f:
        f.write(result['raw_output'])
    print(f"[OK] Log raw salvato in: {raw_file}")


if __name__ == '__main__':
    main()
