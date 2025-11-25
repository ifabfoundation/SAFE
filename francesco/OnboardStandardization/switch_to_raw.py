#!/usr/bin/env python3
"""
Modifica invio.py per usare win_raw.csv invece di win_std_reference.csv
"""
import sys
from pathlib import Path

# Cerca invio.py locale
invio_path = Path("invio.py")

if not invio_path.exists():
    print(f"[ERR] {invio_path} non trovato!")
    sys.exit(1)

content = invio_path.read_text()

# Sostituisci CSV path
if 'CSV  = "win_std_reference.csv"' in content:
    new_content = content.replace('CSV  = "win_std_reference.csv"', 'CSV  = "win_raw.csv"')
    invio_path.write_text(new_content)
    print("✅ invio.py modificato per usare win_raw.csv")
    print("   Ora il firmware riceverà dati RAW e li standardizzerà on-board")
elif 'CSV  = "win_raw.csv"' in content:
    print("✅ invio.py usa già win_raw.csv")
else:
    print("[WARN] Pattern CSV non trovato, controlla manualmente:")
    print(f"       Contenuto trovato: {[line for line in content.split('\\n') if 'CSV' in line]}")
